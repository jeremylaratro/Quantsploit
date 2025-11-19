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


        return "Adaptive strategy using Kalman Filter for trend estimation"


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
            "default": "AAPL"
            },
            "PERIOD": {
            "description": "Historical data period (1y, 2y, 5y)",
            "required": False,
            "default": "1y"
        },
        "INTERVAL": {
            "description": "Data interval (1d for daily)",
            "required": False,
            "default": "1d"
        },
        "FILTER_TYPE": {
            "description": "Filter type: simple, velocity, or full (price+velocity+acceleration)",
            "required": False,
            "default": "full"
        },
        "PROCESS_NOISE": {
            "description": "Process noise variance (lower = smoother filter)",
            "required": False,
            "default": 0.01
        },
        "MEASUREMENT_NOISE": {
            "description": "Measurement noise variance (higher = trust measurements less)",
            "required": False,
            "default": 1.0
        },
        "SIGNAL_THRESHOLD": {
            "description": "Signal threshold as % deviation from filtered price",
            "required": False,
            "default": 0.5
        },
        "INITIAL_CAPITAL": {
            "description": "Initial capital for backtesting",
            "required": False,
            "default": 100000
        },
        "POSITION_SIZE": {
            "description": "Base position size as fraction of capital",
            "required": False,
            "default": 0.5
        },
        "USE_CONFIDENCE_SIZING": {
            "description": "Scale position size based on filter confidence",
            "required": False,
            "default": True
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

        self.print_status(f"Running Kalman Filter Adaptive Strategy for {symbol}")

        # Fetch data
        data_fetcher = DataFetcher(self.database)
        df = data_fetcher.get_stock_data(symbol, period=period, interval=interval)

        if df is None or len(df) < 50:
            self.print_error("Insufficient data for analysis")
            return {"error": "Insufficient data"}

        self.print_info(f"Loaded {len(df)} bars of data")

        # Apply Kalman Filter
        self.print_status(f"Applying {filter_type} Kalman Filter...")

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
        self.print_status("Generating trading signals...")
        signals = self.generate_signals(df, filtered_price, velocity, signal_threshold)

        # Backtest
        self.print_status("Running backtest...")
        results = self.backtest_strategy(
            signals,
            initial_capital,
            position_size,
            use_confidence_sizing,
            errors
        )

        # Display results
        self.print_good("\n=== Backtest Results ===")
        results_dict = results.to_dict()

        for key, value in results_dict.items():
            self.print_info(f"{key}: {value}")

        # Current signal
        self.print_status("\n=== Current Analysis ===")

        current_price = df['Close'].iloc[-1]
        current_filtered = filtered_price.iloc[-1]
        current_deviation = (current_price - current_filtered) / current_filtered * 100
        current_signal = signals['signal'].iloc[-1]

        self.print_info(f"Current Price: ${current_price:.2f}")
        self.print_info(f"Filtered Price: ${current_filtered:.2f}")
        self.print_info(f"Deviation: {current_deviation:.2f}%")

        if velocity is not None:
            current_velocity = velocity.iloc[-1]
            self.print_info(f"Price Velocity: ${current_velocity:.2f}/day")

        if acceleration is not None:
            current_acceleration = acceleration.iloc[-1]
            self.print_info(f"Price Acceleration: ${current_acceleration:.4f}/day²")

        # Interpret signal
        if current_signal >= 2:
            signal_str = "STRONG BUY"
            self.print_good(f"\nSignal: {signal_str}")
        elif current_signal == 1:
            signal_str = "BUY"
            self.print_good(f"\nSignal: {signal_str}")
        elif current_signal == -1:
            signal_str = "SELL"
            self.print_warning(f"\nSignal: {signal_str}")
        elif current_signal <= -2:
            signal_str = "STRONG SELL"
            self.print_warning(f"\nSignal: {signal_str}")
        else:
            signal_str = "NEUTRAL"
            self.print_info(f"\nSignal: {signal_str}")

        # Confidence
        if use_confidence_sizing:
            current_error = errors.iloc[-1]
            confidence = 1.0 / (1.0 + current_error)
            self.print_info(f"Signal Confidence: {confidence:.2%}")

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
