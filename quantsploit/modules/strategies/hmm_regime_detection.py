"""
Hidden Markov Model (HMM) Regime Detection Strategy

This strategy uses Hidden Markov Models to:
- Detect different market regimes (bull, bear, sideways/volatile)
- Adapt trading strategies based on current regime
- Predict regime transitions
- Generate regime-aware trading signals

State-of-the-art techniques:
- Gaussian HMM with regime-specific parameters
- Multi-feature regime classification
- Regime transition probability analysis
- Adaptive strategy selection based on regime
"""

from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher
from quantsploit.utils.ta_compat import rsi, atr, adx
from quantsploit.utils.backtesting import (
    Backtester, BacktestConfig, BacktestResults
)
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HMMRegimeDetectionStrategy(BaseModule):
    """
    Hidden Markov Model Regime Detection Strategy

    Uses HMM to detect market regimes and adapt trading accordingly
    """

    @property
    def name(self) -> str:


        return "hmm_regime_detection"


    @property
    def description(self) -> str:


        return "Market regime detection using Hidden Markov Models"


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
            "value": "SPY"
            },
            "PERIOD": {
            "description": "Historical data period (2y, 5y, max)",
            "required": False,
            "value": "2y"
        },
        "INTERVAL": {
            "description": "Data interval (1d for daily)",
            "required": False,
            "value": "1d"
        },
        "NUM_REGIMES": {
            "description": "Number of market regimes to detect (2-5)",
            "required": False,
            "value": 3
        },
        "LOOKBACK": {
            "description": "Lookback period for regime features",
            "required": False,
            "value": 20
        },
        "INITIAL_CAPITAL": {
            "description": "Initial capital for backtesting",
            "required": False,
            "value": 100000
        },
        "POSITION_SIZE": {
            "description": "Position size as fraction of capital",
            "required": False,
            "value": 0.5
        },
        "BULL_STRATEGY": {
            "description": "Strategy in bull regime: trend_follow, mean_revert, or hold",
            "required": False,
            "value": "trend_follow"
        },
        "BEAR_STRATEGY": {
            "description": "Strategy in bear regime: defensive, short, or cash",
            "required": False,
            "value": "defensive"
        },
        "SIDEWAYS_STRATEGY": {
            "description": "Strategy in sideways regime: mean_revert or wait",
            "required": False,
            "value": "mean_revert"
        }
        })

    def extract_regime_features(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """
        Extract features for regime detection

        Returns:
            DataFrame with features: returns, volatility, trend
        """
        features = pd.DataFrame(index=df.index)

        # Returns at different timeframes
        features['returns_1d'] = df['Close'].pct_change(1)
        features['returns_5d'] = df['Close'].pct_change(5)
        features['returns_20d'] = df['Close'].pct_change(20)

        # Volatility (rolling std of returns)
        features['volatility'] = df['Close'].pct_change().rolling(lookback).std()

        # Trend strength using linear regression slope
        def calc_trend(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            y = series.values
            # Simple linear regression
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            return slope

        features['trend'] = df['Close'].rolling(lookback).apply(calc_trend, raw=False)

        # Volume trend
        features['volume_change'] = df['Volume'].pct_change(lookback)

        # RSI
        rsi_data = rsi(df, 14)
        features['rsi'] = rsi_data['RSI_14']

        # ATR (volatility measure)
        atr_data = atr(df, 14)
        features['atr'] = atr_data['ATR_14']
        features['atr_pct'] = features['atr'] / df['Close']

        # ADX (trend strength)
        adx_data = adx(df)
        features['adx'] = adx_data['ADX']

        return features

    def simple_hmm(
        self,
        observations: np.ndarray,
        n_states: int,
        n_iter: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simple Gaussian HMM implementation using EM algorithm

        Args:
            observations: Observation sequence (T x D)
            n_states: Number of hidden states
            n_iter: Number of EM iterations

        Returns:
            Tuple of (states, transition_matrix, means, covariances)
        """
        T, D = observations.shape

        # Initialize parameters randomly
        np.random.seed(42)

        # Initial state probabilities (uniform)
        pi = np.ones(n_states) / n_states

        # Transition matrix (slightly favor staying in same state)
        A = np.random.rand(n_states, n_states)
        for i in range(n_states):
            A[i, i] += 1.0  # Favor staying
        A = A / A.sum(axis=1, keepdims=True)

        # Emission parameters (Gaussian)
        # Initialize means by k-means-like clustering
        indices = np.random.choice(T, n_states, replace=False)
        means = observations[indices].copy()

        # Initialize covariances
        covs = np.array([np.eye(D) for _ in range(n_states)])

        # EM algorithm
        for iteration in range(n_iter):
            # E-step: Forward-Backward algorithm (simplified)
            # For simplicity, we'll use Viterbi decoding instead of full forward-backward

            # Compute emission probabilities
            B = np.zeros((T, n_states))
            for t in range(T):
                for s in range(n_states):
                    diff = observations[t] - means[s]
                    cov_inv = np.linalg.inv(covs[s] + 1e-6 * np.eye(D))
                    det = np.linalg.det(covs[s] + 1e-6 * np.eye(D))
                    B[t, s] = np.exp(-0.5 * diff.T @ cov_inv @ diff) / np.sqrt((2 * np.pi) ** D * det)

            # Viterbi algorithm for state sequence
            delta = np.zeros((T, n_states))
            psi = np.zeros((T, n_states), dtype=int)

            # Initialize
            delta[0] = pi * B[0]

            # Recursion
            for t in range(1, T):
                for s in range(n_states):
                    probs = delta[t - 1] * A[:, s]
                    psi[t, s] = np.argmax(probs)
                    delta[t, s] = np.max(probs) * B[t, s]

            # Backtrack to find optimal state sequence
            states = np.zeros(T, dtype=int)
            states[T - 1] = np.argmax(delta[T - 1])
            for t in range(T - 2, -1, -1):
                states[t] = psi[t + 1, states[t + 1]]

            # M-step: Update parameters
            for s in range(n_states):
                mask = (states == s)
                if mask.sum() > 0:
                    # Update mean
                    means[s] = observations[mask].mean(axis=0)

                    # Update covariance
                    diff = observations[mask] - means[s]
                    covs[s] = (diff.T @ diff) / mask.sum()

            # Update transition matrix
            for i in range(n_states):
                for j in range(n_states):
                    transitions = ((states[:-1] == i) & (states[1:] == j)).sum()
                    A[i, j] = (transitions + 1e-6) / (mask.sum() + n_states * 1e-6)

            A = A / A.sum(axis=1, keepdims=True)

        return states, A, means, covs

    def classify_regimes(
        self,
        states: np.ndarray,
        features: pd.DataFrame,
        n_states: int
    ) -> Tuple[Dict, np.ndarray]:
        """
        Classify detected states into interpretable regimes

        Args:
            states: HMM state sequence
            features: Feature DataFrame
            n_states: Number of states

        Returns:
            Tuple of (regime_mapping, regime_sequence)
        """
        # Calculate characteristics of each state
        state_chars = {}

        for s in range(n_states):
            mask = (states == s)
            if mask.sum() > 0:
                avg_return = features.loc[mask, 'returns_20d'].mean()
                avg_volatility = features.loc[mask, 'volatility'].mean()
                avg_trend = features.loc[mask, 'trend'].mean()

                state_chars[s] = {
                    'return': avg_return,
                    'volatility': avg_volatility,
                    'trend': avg_trend
                }

        # Map states to regimes based on characteristics
        regime_mapping = {}

        # Find bull regime (highest returns, positive trend)
        bull_state = max(state_chars.keys(),
                        key=lambda s: state_chars[s]['return'] + state_chars[s]['trend'])
        regime_mapping[bull_state] = 'BULL'

        # Find bear regime (lowest returns, negative trend)
        bear_state = min(state_chars.keys(),
                        key=lambda s: state_chars[s]['return'] + state_chars[s]['trend'])
        regime_mapping[bear_state] = 'BEAR'

        # Remaining states are sideways/volatile
        for s in state_chars.keys():
            if s not in regime_mapping:
                if state_chars[s]['volatility'] > np.median([state_chars[x]['volatility'] for x in state_chars.keys()]):
                    regime_mapping[s] = 'VOLATILE'
                else:
                    regime_mapping[s] = 'SIDEWAYS'

        # Map state sequence to regimes
        regime_sequence = np.array([regime_mapping[s] for s in states])

        return regime_mapping, regime_sequence

    def generate_regime_signals(
        self,
        df: pd.DataFrame,
        regimes: np.ndarray,
        bull_strategy: str,
        bear_strategy: str,
        sideways_strategy: str,
        features: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate trading signals based on current regime

        Args:
            df: Price data
            regimes: Regime sequence
            bull_strategy: Strategy for bull regime
            bear_strategy: Strategy for bear regime
            sideways_strategy: Strategy for sideways regime
            features: Feature DataFrame

        Returns:
            DataFrame with signals
        """
        signals = df.copy()
        signals['regime'] = regimes
        signals['signal'] = 0

        for i in range(len(signals)):
            regime = regimes[i]

            if regime == 'BULL':
                if bull_strategy == 'trend_follow':
                    # Buy on dips, hold on strength
                    if i > 0 and features.iloc[i]['returns_1d'] < 0:
                        signals.iloc[i, signals.columns.get_loc('signal')] = 1  # Buy
                    else:
                        signals.iloc[i, signals.columns.get_loc('signal')] = 0  # Hold

                elif bull_strategy == 'mean_revert':
                    # Buy oversold
                    if features.iloc[i]['rsi'] < 30:
                        signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    elif features.iloc[i]['rsi'] > 70:
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1

                else:  # hold
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0

            elif regime == 'BEAR':
                if bear_strategy == 'defensive':
                    # Exit positions, go to cash
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1

                elif bear_strategy == 'short':
                    # Short on rallies
                    if i > 0 and features.iloc[i]['returns_1d'] > 0:
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    else:
                        signals.iloc[i, signals.columns.get_loc('signal')] = 0

                else:  # cash
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1

            elif regime in ['SIDEWAYS', 'VOLATILE']:
                if sideways_strategy == 'mean_revert':
                    # Mean reversion in range
                    if features.iloc[i]['rsi'] < 30:
                        signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    elif features.iloc[i]['rsi'] > 70:
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    else:
                        signals.iloc[i, signals.columns.get_loc('signal')] = 0

                else:  # wait
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0

        return signals

    def backtest_strategy(
        self,
        signals: pd.DataFrame,
        initial_capital: float,
        position_size: float
    ) -> BacktestResults:
        """Backtest the HMM regime strategy"""
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
        """Execute the HMM regime detection strategy"""
        symbol = self.options["SYMBOL"]["value"]
        period = self.options["PERIOD"]["value"]
        interval = self.options["INTERVAL"]["value"]
        num_regimes = int(self.options["NUM_REGIMES"]["value"])
        lookback = int(self.options["LOOKBACK"]["value"])
        initial_capital = float(self.options["INITIAL_CAPITAL"]["value"])
        position_size = float(self.options["POSITION_SIZE"]["value"])
        bull_strategy = self.options["BULL_STRATEGY"]["value"]
        bear_strategy = self.options["BEAR_STRATEGY"]["value"]
        sideways_strategy = self.options["SIDEWAYS_STRATEGY"]["value"]

        self.print_status(f"Running HMM Regime Detection for {symbol}")

        # Fetch data
        data_fetcher = DataFetcher(self.database)
        df = data_fetcher.get_stock_data(symbol, period=period, interval=interval)

        if df is None or len(df) < 100:
            self.print_error("Insufficient data for analysis")
            return {"error": "Insufficient data"}

        self.print_info(f"Loaded {len(df)} bars of data")

        # Extract features
        self.print_status("Extracting regime features...")
        features = self.extract_regime_features(df, lookback)
        features = features.dropna()

        # Prepare observations for HMM
        feature_cols = ['returns_20d', 'volatility', 'trend']
        observations = features[feature_cols].values

        # Normalize features
        obs_mean = observations.mean(axis=0)
        obs_std = observations.std(axis=0)
        observations_norm = (observations - obs_mean) / (obs_std + 1e-8)

        # Fit HMM
        self.print_status(f"Fitting HMM with {num_regimes} regimes...")
        states, transition_matrix, means, covs = self.simple_hmm(
            observations_norm,
            num_regimes,
            n_iter=50
        )

        # Classify regimes
        regime_mapping, regimes = self.classify_regimes(states, features, num_regimes)

        self.print_good("\n=== Regime Mapping ===")
        for state, regime in regime_mapping.items():
            self.print_info(f"State {state} -> {regime}")

        # Regime statistics
        self.print_good("\n=== Regime Statistics ===")
        for regime_name in set(regimes):
            mask = (regimes == regime_name)
            count = mask.sum()
            pct = count / len(regimes) * 100

            regime_returns = features.loc[mask, 'returns_20d'].mean() * 100
            regime_vol = features.loc[mask, 'volatility'].mean() * 100

            self.print_info(f"{regime_name}:")
            self.print_info(f"  Frequency: {count} bars ({pct:.1f}%)")
            self.print_info(f"  Avg 20d Return: {regime_returns:.2f}%")
            self.print_info(f"  Avg Volatility: {regime_vol:.2f}%")

        # Transition probabilities
        self.print_good("\n=== Transition Probabilities ===")
        for i, regime_i in regime_mapping.items():
            self.print_info(f"From {regime_i}:")
            for j, regime_j in regime_mapping.items():
                prob = transition_matrix[i, j]
                self.print_info(f"  To {regime_j}: {prob:.2%}")

        # Generate signals
        self.print_status("\nGenerating regime-based signals...")

        # Align regimes with original dataframe
        aligned_regimes = pd.Series(index=df.index, dtype=object)
        aligned_regimes[features.index] = regimes

        signals = self.generate_regime_signals(
            df,
            aligned_regimes.fillna('UNKNOWN'),
            bull_strategy,
            bear_strategy,
            sideways_strategy,
            features.reindex(df.index)
        )

        # Backtest
        self.print_status("Running backtest...")
        results = self.backtest_strategy(signals, initial_capital, position_size)

        # Display results
        self.print_good("\n=== Backtest Results ===")
        results_dict = results.to_dict()

        for key, value in results_dict.items():
            self.print_info(f"{key}: {value}")

        # Current regime
        self.print_status("\n=== Current Market Regime ===")
        current_regime = regimes[-1]
        current_price = df['Close'].iloc[-1]

        self.print_good(f"Current Regime: {current_regime}")
        self.print_info(f"Current Price: ${current_price:.2f}")

        # Regime transition probabilities
        current_state = states[-1]
        self.print_info("\nTransition Probabilities:")
        for state, regime in regime_mapping.items():
            prob = transition_matrix[current_state, state]
            self.print_info(f"  To {regime}: {prob:.2%}")

        # Trading recommendation
        current_signal = signals['signal'].iloc[-1]
        if current_signal == 1:
            self.print_good("\nRecommendation: BUY")
        elif current_signal == -1:
            self.print_warning("\nRecommendation: SELL/EXIT")
        else:
            self.print_info("\nRecommendation: HOLD/WAIT")

        return {
            "symbol": symbol,
            "current_regime": current_regime,
            "current_price": float(current_price),
            "recommendation": "BUY" if current_signal == 1 else "SELL" if current_signal == -1 else "HOLD",
            "regime_stats": {
                regime: {
                    "frequency": float((regimes == regime).sum() / len(regimes)),
                    "avg_return": float(features.loc[regimes == regime, 'returns_20d'].mean())
                }
                for regime in set(regimes)
            },
            "backtest_results": results_dict,
            "total_return_pct": results.total_return_pct,
            "sharpe_ratio": results.sharpe_ratio,
            "win_rate": results.win_rate
        }
