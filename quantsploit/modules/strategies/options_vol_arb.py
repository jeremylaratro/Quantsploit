"""
Options Volatility Arbitrage Strategy for Quantsploit

This module implements options volatility arbitrage strategies that exploit
discrepancies between implied volatility and realized/forecast volatility.

Key Features:
- Implied vs realized volatility comparison
- Volatility surface arbitrage detection
- Delta-neutral position construction
- Gamma scalping simulation
- Variance swap replication
- Integration with backtesting framework

References:
    - Sinclair, E. (2008). "Volatility Trading"
    - Bennett, C. (2014). "Trading Volatility"
    - Gatheral, J. (2006). "The Volatility Surface"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.stats import norm

logger = logging.getLogger(__name__)


class VolArbitrageType(Enum):
    """Types of volatility arbitrage strategies."""
    LONG_VOL = "long_volatility"
    SHORT_VOL = "short_volatility"
    CALENDAR_SPREAD = "calendar_spread"
    SKEW_TRADE = "skew_trade"
    TERM_STRUCTURE = "term_structure"


@dataclass
class VolatilitySignal:
    """
    Volatility arbitrage trading signal.

    Attributes:
        date: Signal date
        symbol: Underlying symbol
        signal_type: Type of vol arbitrage
        direction: 'long_vol' or 'short_vol'
        implied_vol: Current implied volatility
        forecast_vol: Forecasted realized volatility
        vol_spread: IV - forecast spread
        confidence: Signal confidence (0-1)
        structure: Option structure to trade
    """
    date: pd.Timestamp
    symbol: str
    signal_type: VolArbitrageType
    direction: str
    implied_vol: float
    forecast_vol: float
    vol_spread: float
    confidence: float
    structure: Dict = field(default_factory=dict)


@dataclass
class VolPosition:
    """
    Volatility arbitrage position.

    Attributes:
        symbol: Underlying symbol
        entry_date: Position entry date
        entry_iv: Entry implied volatility
        entry_price: Underlying price at entry
        structure: Options positions
        delta: Current position delta
        gamma: Current position gamma
        vega: Current position vega
        theta: Current position theta
    """
    symbol: str
    entry_date: pd.Timestamp
    entry_iv: float
    entry_price: float
    structure: Dict
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0


class OptionsVolatilityArbitrage:
    """
    Options Volatility Arbitrage Strategy.

    Identifies and trades discrepancies between implied and realized volatility.
    Constructs delta-neutral positions to isolate volatility exposure.

    ★ Insight ─────────────────────────────────────
    Volatility Arbitrage Key Concepts:
    - IV > Realized Vol = Sell vol (short straddles/strangles)
    - IV < Realized Vol = Buy vol (long straddles/gamma scalp)
    - Delta hedging isolates pure volatility exposure
    - Theta decay funds gamma profits (or vice versa)
    - Vol mean reverts; extreme readings offer best opportunities
    ─────────────────────────────────────────────────

    Example:
        >>> strategy = OptionsVolatilityArbitrage(price_data, iv_data)
        >>> signals = strategy.generate_signals()
        >>> results = strategy.run_backtest()

    Attributes:
        price_data: DataFrame of underlying prices
        iv_data: DataFrame of implied volatilities
        options_chain: Optional detailed options chain data
    """

    def __init__(
        self,
        price_data: pd.DataFrame,
        iv_data: Optional[pd.DataFrame] = None,
        options_chain: Optional[pd.DataFrame] = None,
        vol_forecast_window: int = 20,
        vol_lookback: int = 252,
        min_vol_spread: float = 0.03,  # 3% minimum spread
        risk_free_rate: float = 0.05
    ):
        """
        Initialize Options Volatility Arbitrage Strategy.

        Args:
            price_data: DataFrame with OHLCV data
            iv_data: DataFrame of implied volatilities (ATM or by strike)
            options_chain: Full options chain data (optional)
            vol_forecast_window: Days for realized vol calculation
            vol_lookback: Days for vol statistics
            min_vol_spread: Minimum IV-RV spread for signal
            risk_free_rate: Risk-free rate for BS calculations
        """
        self.price_data = price_data.copy()
        self.iv_data = iv_data.copy() if iv_data is not None else None
        self.options_chain = options_chain
        self.vol_forecast_window = vol_forecast_window
        self.vol_lookback = vol_lookback
        self.min_vol_spread = min_vol_spread
        self.risk_free_rate = risk_free_rate

        # Calculate realized volatility
        self._calculate_realized_vol()

        # Calculate vol statistics
        self._calculate_vol_statistics()

    def _calculate_realized_vol(self) -> None:
        """Calculate historical realized volatility."""
        if 'Close' in self.price_data.columns:
            prices = self.price_data['Close']
        else:
            prices = self.price_data.iloc[:, 0]

        # Log returns
        returns = np.log(prices / prices.shift(1))

        # Rolling realized vol (annualized)
        self.realized_vol = returns.rolling(self.vol_forecast_window).std() * np.sqrt(252)

        # Multiple windows for analysis
        self.vol_10d = returns.rolling(10).std() * np.sqrt(252)
        self.vol_20d = returns.rolling(20).std() * np.sqrt(252)
        self.vol_60d = returns.rolling(60).std() * np.sqrt(252)

    def _calculate_vol_statistics(self) -> None:
        """Calculate volatility statistics for percentile ranking."""
        self.vol_mean = self.realized_vol.rolling(self.vol_lookback).mean()
        self.vol_std = self.realized_vol.rolling(self.vol_lookback).std()

        # Percentile ranking
        self.vol_percentile = self.realized_vol.rolling(self.vol_lookback).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() * 100 if len(x) > 1 else 50
        )

    def forecast_volatility(
        self,
        method: str = 'ewma',
        as_of_date: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Forecast future realized volatility.

        Args:
            method: Forecast method ('ewma', 'garch', 'simple')
            as_of_date: Forecast as of this date

        Returns:
            Forecasted volatility (annualized)
        """
        if 'Close' in self.price_data.columns:
            prices = self.price_data['Close']
        else:
            prices = self.price_data.iloc[:, 0]

        if as_of_date is not None:
            prices = prices.loc[:as_of_date]

        returns = np.log(prices / prices.shift(1)).dropna()

        if len(returns) < self.vol_forecast_window:
            return 0.20  # Default

        if method == 'simple':
            # Simple historical vol
            return returns.tail(self.vol_forecast_window).std() * np.sqrt(252)

        elif method == 'ewma':
            # Exponentially weighted MA
            lambda_decay = 0.94  # RiskMetrics standard
            weights = np.array([(1 - lambda_decay) * (lambda_decay ** i)
                               for i in range(len(returns))])[::-1]
            weights = weights / weights.sum()

            weighted_var = np.sum(weights * (returns ** 2))
            return np.sqrt(weighted_var * 252)

        elif method == 'garch':
            # Simple GARCH(1,1) approximation
            omega = 0.000001
            alpha = 0.05
            beta = 0.94

            variance = returns.var()
            for r in returns.tail(60):
                variance = omega + alpha * (r ** 2) + beta * variance

            return np.sqrt(variance * 252)

        else:
            return returns.tail(self.vol_forecast_window).std() * np.sqrt(252)

    def calculate_vol_cone(
        self,
        windows: List[int] = [10, 20, 30, 60, 90, 120],
        as_of_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Calculate volatility cone (percentile bands by horizon).

        Args:
            windows: List of lookback windows
            as_of_date: Calculate as of this date

        Returns:
            DataFrame with percentile bands for each window
        """
        if 'Close' in self.price_data.columns:
            prices = self.price_data['Close']
        else:
            prices = self.price_data.iloc[:, 0]

        if as_of_date is not None:
            prices = prices.loc[:as_of_date]

        returns = np.log(prices / prices.shift(1)).dropna()

        cone_data = []
        for window in windows:
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            rolling_vol = rolling_vol.dropna()

            if len(rolling_vol) == 0:
                continue

            cone_data.append({
                'window': window,
                'current': rolling_vol.iloc[-1],
                'p10': rolling_vol.quantile(0.10),
                'p25': rolling_vol.quantile(0.25),
                'median': rolling_vol.quantile(0.50),
                'p75': rolling_vol.quantile(0.75),
                'p90': rolling_vol.quantile(0.90),
                'min': rolling_vol.min(),
                'max': rolling_vol.max()
            })

        return pd.DataFrame(cone_data)

    def detect_vol_mispricing(
        self,
        symbol: str,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> Optional[VolatilitySignal]:
        """
        Detect volatility mispricing (IV vs forecast RV).

        Args:
            symbol: Underlying symbol
            as_of_date: Detect as of this date

        Returns:
            VolatilitySignal if mispricing detected, else None
        """
        if self.iv_data is None:
            return None

        if as_of_date is None:
            as_of_date = self.price_data.index[-1]

        # Get implied vol
        try:
            if symbol in self.iv_data.columns:
                iv = self.iv_data.loc[:as_of_date, symbol].iloc[-1]
            else:
                iv = self.iv_data.loc[:as_of_date].iloc[-1, 0]
        except Exception:
            return None

        # Forecast realized vol
        forecast_vol = self.forecast_volatility('ewma', as_of_date)

        # Calculate spread
        vol_spread = iv - forecast_vol

        # Check if spread is significant
        if abs(vol_spread) < self.min_vol_spread:
            return None

        # Calculate confidence based on percentile extremes
        vol_cone = self.calculate_vol_cone(as_of_date=as_of_date)
        if len(vol_cone) == 0:
            return None

        current_20d = vol_cone[vol_cone['window'] == 20]['current'].values
        if len(current_20d) == 0:
            return None

        current_20d = current_20d[0]
        median_20d = vol_cone[vol_cone['window'] == 20]['median'].values[0]
        p90_20d = vol_cone[vol_cone['window'] == 20]['p90'].values[0]
        p10_20d = vol_cone[vol_cone['window'] == 20]['p10'].values[0]

        # Confidence based on how extreme the reading is
        if vol_spread > 0:  # IV > RV, sell vol
            direction = 'short_vol'
            # Higher confidence if IV is very high relative to history
            confidence = min((iv - median_20d) / (p90_20d - median_20d), 1.0) if p90_20d > median_20d else 0.5
        else:  # IV < RV, buy vol
            direction = 'long_vol'
            confidence = min((median_20d - iv) / (median_20d - p10_20d), 1.0) if median_20d > p10_20d else 0.5

        confidence = max(0.3, min(confidence, 1.0))

        # Determine structure
        if direction == 'short_vol':
            structure = self._short_vol_structure(iv, forecast_vol)
        else:
            structure = self._long_vol_structure(iv, forecast_vol)

        return VolatilitySignal(
            date=as_of_date,
            symbol=symbol,
            signal_type=VolArbitrageType.SHORT_VOL if direction == 'short_vol' else VolArbitrageType.LONG_VOL,
            direction=direction,
            implied_vol=iv,
            forecast_vol=forecast_vol,
            vol_spread=vol_spread,
            confidence=confidence,
            structure=structure
        )

    def _short_vol_structure(
        self,
        iv: float,
        forecast_vol: float
    ) -> Dict:
        """Generate short volatility structure."""
        # Prefer iron condor for defined risk, straddle for max premium
        spread = iv - forecast_vol

        if spread > 0.10:  # Very high IV
            return {
                'strategy': 'short_straddle',
                'description': 'Sell ATM straddle for maximum premium',
                'delta_hedge': True,
                'target_delta': 0.0
            }
        else:
            return {
                'strategy': 'iron_condor',
                'description': 'Sell iron condor for defined risk premium',
                'delta_hedge': False,
                'wing_width': 0.10  # 10% OTM wings
            }

    def _long_vol_structure(
        self,
        iv: float,
        forecast_vol: float
    ) -> Dict:
        """Generate long volatility structure."""
        spread = forecast_vol - iv

        if spread > 0.05:  # Significantly cheap vol
            return {
                'strategy': 'long_straddle',
                'description': 'Buy ATM straddle for gamma exposure',
                'gamma_scalp': True,
                'scalp_threshold': 0.01  # Delta threshold for hedging
            }
        else:
            return {
                'strategy': 'long_strangle',
                'description': 'Buy OTM strangle for convexity',
                'gamma_scalp': False,
                'strike_offset': 0.05  # 5% OTM
            }

    def generate_signals(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> List[VolatilitySignal]:
        """
        Generate volatility arbitrage signals.

        Args:
            symbols: List of symbols to analyze
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            List of VolatilitySignal objects
        """
        if self.iv_data is None:
            logger.warning("No IV data provided, cannot generate signals")
            return []

        if symbols is None:
            if isinstance(self.iv_data.columns, pd.MultiIndex):
                symbols = self.iv_data.columns.get_level_values(0).unique().tolist()
            else:
                symbols = self.iv_data.columns.tolist()

        if start_date is None:
            start_date = self.price_data.index[self.vol_lookback]
        if end_date is None:
            end_date = self.price_data.index[-1]

        signals = []

        # Generate signals periodically (weekly)
        dates = self.price_data.loc[start_date:end_date].index[::5]

        for date in dates:
            for symbol in symbols:
                signal = self.detect_vol_mispricing(symbol, date)
                if signal is not None:
                    signals.append(signal)

        return signals

    def calculate_gamma_scalp_pnl(
        self,
        entry_iv: float,
        entry_price: float,
        price_series: pd.Series,
        days_to_expiry: int = 30,
        scalp_threshold: float = 0.01,
        contracts: int = 1
    ) -> Dict:
        """
        Simulate gamma scalping P&L.

        Args:
            entry_iv: Entry implied volatility
            entry_price: Entry underlying price
            price_series: Series of underlying prices
            days_to_expiry: Days until option expiry
            scalp_threshold: Delta threshold for rehedging
            contracts: Number of straddle contracts

        Returns:
            Dictionary with gamma scalping results
        """
        # Straddle initial values
        T = days_to_expiry / 365
        call_delta = 0.5  # ATM
        put_delta = -0.5

        # Position: long 1 ATM straddle
        net_delta = call_delta + put_delta  # ~0

        # Greeks (approximate)
        # Gamma for ATM straddle
        d1 = 0  # ATM
        gamma = norm.pdf(d1) / (entry_price * entry_iv * np.sqrt(T))

        # Theta (decay)
        theta_daily = -entry_price * entry_iv * norm.pdf(d1) / (2 * np.sqrt(T)) / 365

        # Premium paid (approximate Black-Scholes)
        premium = 2 * entry_price * entry_iv * np.sqrt(T) * 0.4 * contracts

        # Simulate
        hedge_shares = 0
        hedge_pnl = 0
        theta_cost = 0
        gamma_profits = 0

        prev_price = entry_price
        remaining_days = days_to_expiry

        for i, price in enumerate(price_series):
            if remaining_days <= 0:
                break

            # Time decay
            T_remaining = remaining_days / 365
            theta_cost += abs(theta_daily) * contracts

            # Price movement
            move = price - prev_price

            # Gamma profit from price movement
            gamma_profit = 0.5 * gamma * (move ** 2) * contracts * 100
            gamma_profits += gamma_profit

            # Update delta
            # Simplified: delta changes with price
            if T_remaining > 0:
                moneyness = price / entry_price
                new_call_delta = norm.cdf((np.log(moneyness) + 0.5 * (entry_iv ** 2) * T_remaining) /
                                         (entry_iv * np.sqrt(T_remaining)))
                new_put_delta = new_call_delta - 1
                net_delta = (new_call_delta + new_put_delta) * contracts * 100

            # Check if rehedge needed
            if abs(net_delta + hedge_shares) > scalp_threshold * 100 * contracts:
                # Rehedge
                target_hedge = -int(net_delta)
                shares_to_trade = target_hedge - hedge_shares
                hedge_pnl -= shares_to_trade * price * 0.001  # Transaction cost
                hedge_shares = target_hedge

            # Update hedge P&L from price change
            hedge_pnl += hedge_shares * move

            prev_price = price
            remaining_days -= 1

        # Final P&L
        total_pnl = gamma_profits + hedge_pnl - theta_cost - premium

        return {
            'premium_paid': premium,
            'gamma_profits': gamma_profits,
            'hedge_pnl': hedge_pnl,
            'theta_cost': theta_cost,
            'total_pnl': total_pnl,
            'realized_vol': price_series.pct_change().std() * np.sqrt(252),
            'entry_iv': entry_iv
        }

    def run_backtest(
        self,
        initial_capital: float = 100000.0,
        position_size_pct: float = 0.05,
        holding_period: int = 30,
        commission_per_contract: float = 1.0,
        symbols: Optional[List[str]] = None
    ) -> Dict:
        """
        Backtest volatility arbitrage strategy.

        Args:
            initial_capital: Starting capital
            position_size_pct: Position size as % of capital
            holding_period: Days to hold position
            commission_per_contract: Commission per option contract
            symbols: Symbols to trade

        Returns:
            Dictionary with backtest results
        """
        signals = self.generate_signals(symbols=symbols)

        if len(signals) == 0:
            return {
                'error': 'No signals generated',
                'n_signals': 0
            }

        capital = initial_capital
        positions = []  # Active positions
        trades = []

        # Process signals chronologically
        signals = sorted(signals, key=lambda x: x.date)

        # Get all unique dates
        all_dates = self.price_data.index

        signal_dict = {}
        for s in signals:
            if s.date not in signal_dict:
                signal_dict[s.date] = []
            signal_dict[s.date].append(s)

        for i, date in enumerate(all_dates):
            # Check for position exits
            positions_to_close = []
            for pos_idx, pos in enumerate(positions):
                days_held = (date - pos['entry_date']).days
                if days_held >= holding_period:
                    positions_to_close.append(pos_idx)

            # Close positions
            for pos_idx in reversed(positions_to_close):
                pos = positions[pos_idx]

                # Calculate P&L based on vol change
                try:
                    current_iv = self.iv_data.loc[date, pos['symbol']] if self.iv_data is not None else pos['entry_iv']
                except Exception:
                    current_iv = pos['entry_iv']

                iv_change = current_iv - pos['entry_iv']

                # P&L from vega (simplified)
                if pos['direction'] == 'short_vol':
                    pnl = -iv_change * pos['vega'] * pos['notional']
                else:
                    pnl = iv_change * pos['vega'] * pos['notional']

                # Add theta earned/paid
                days = (date - pos['entry_date']).days
                if pos['direction'] == 'short_vol':
                    pnl += pos['theta'] * days * pos['notional']  # Earn theta
                else:
                    pnl -= pos['theta'] * days * pos['notional']  # Pay theta

                capital += pnl

                trades.append({
                    'symbol': pos['symbol'],
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'direction': pos['direction'],
                    'entry_iv': pos['entry_iv'],
                    'exit_iv': current_iv,
                    'iv_change': iv_change,
                    'pnl': pnl
                })

                del positions[pos_idx]

            # Check for new signals
            if date in signal_dict:
                for signal in signal_dict[date]:
                    # Check if already have position in this symbol
                    existing = [p for p in positions if p['symbol'] == signal.symbol]
                    if len(existing) > 0:
                        continue

                    # Size position
                    position_value = capital * position_size_pct

                    # Approximate vega and theta for ATM straddle
                    # Vega ~ 0.4 * S * sqrt(T) for ATM straddle
                    try:
                        if 'Close' in self.price_data.columns:
                            spot = self.price_data.loc[date, 'Close']
                        else:
                            spot = self.price_data.loc[date].iloc[0]
                    except Exception:
                        continue

                    T = holding_period / 365
                    vega = 0.4 * spot * np.sqrt(T) / 100  # Per 1% IV change
                    theta = spot * signal.implied_vol / (365 * 2 * np.sqrt(T)) / 100  # Daily decay

                    # Number of contracts based on position size
                    contracts = int(position_value / (spot * 0.1))  # Rough premium estimate

                    positions.append({
                        'symbol': signal.symbol,
                        'entry_date': date,
                        'entry_iv': signal.implied_vol,
                        'forecast_vol': signal.forecast_vol,
                        'direction': signal.direction,
                        'notional': contracts,
                        'vega': vega,
                        'theta': theta
                    })

        # Calculate metrics
        if len(trades) == 0:
            return {
                'error': 'No completed trades',
                'n_signals': len(signals)
            }

        trades_df = pd.DataFrame(trades)
        total_return = (capital - initial_capital) / initial_capital
        win_rate = (trades_df['pnl'] > 0).mean()

        # Separate long vol vs short vol performance
        long_vol_trades = trades_df[trades_df['direction'] == 'long_vol']
        short_vol_trades = trades_df[trades_df['direction'] == 'short_vol']

        return {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return_pct': total_return * 100,
            'n_signals': len(signals),
            'n_trades': len(trades),
            'win_rate': win_rate,
            'avg_trade_pnl': trades_df['pnl'].mean(),
            'long_vol_trades': len(long_vol_trades),
            'short_vol_trades': len(short_vol_trades),
            'long_vol_winrate': (long_vol_trades['pnl'] > 0).mean() if len(long_vol_trades) > 0 else 0,
            'short_vol_winrate': (short_vol_trades['pnl'] > 0).mean() if len(short_vol_trades) > 0 else 0,
            'trades': trades
        }

    def analyze_term_structure(
        self,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> Dict:
        """
        Analyze volatility term structure.

        Args:
            as_of_date: Analyze as of this date

        Returns:
            Term structure analysis
        """
        if self.options_chain is None:
            return {'error': 'No options chain data'}

        # Group by expiration
        expirations = self.options_chain['expiration'].unique()
        expirations = sorted(expirations)

        term_structure = []
        for exp in expirations:
            exp_options = self.options_chain[self.options_chain['expiration'] == exp]
            atm_options = exp_options[abs(exp_options['moneyness'] - 1.0) < 0.05]

            if len(atm_options) > 0:
                avg_iv = atm_options['implied_vol'].mean()
                days_to_exp = (exp - as_of_date).days if as_of_date else 30

                term_structure.append({
                    'expiration': exp,
                    'days_to_expiry': days_to_exp,
                    'atm_iv': avg_iv
                })

        if len(term_structure) < 2:
            return {'error': 'Insufficient term structure data'}

        ts_df = pd.DataFrame(term_structure)

        # Analyze shape
        short_term_iv = ts_df.iloc[0]['atm_iv']
        long_term_iv = ts_df.iloc[-1]['atm_iv']

        if short_term_iv > long_term_iv * 1.05:
            shape = 'backwardation'  # Near > Far, usually fear
        elif long_term_iv > short_term_iv * 1.05:
            shape = 'contango'  # Far > Near, normal
        else:
            shape = 'flat'

        return {
            'term_structure': ts_df,
            'shape': shape,
            'short_term_iv': short_term_iv,
            'long_term_iv': long_term_iv,
            'spread': short_term_iv - long_term_iv
        }

    def analyze_skew(
        self,
        expiration: pd.Timestamp,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> Dict:
        """
        Analyze volatility skew for an expiration.

        Args:
            expiration: Options expiration date
            as_of_date: Analyze as of this date

        Returns:
            Skew analysis
        """
        if self.options_chain is None:
            return {'error': 'No options chain data'}

        exp_options = self.options_chain[self.options_chain['expiration'] == expiration]

        if len(exp_options) < 5:
            return {'error': 'Insufficient skew data'}

        # Sort by strike
        exp_options = exp_options.sort_values('strike')

        # Get puts and calls
        puts = exp_options[exp_options['option_type'] == 'put']
        calls = exp_options[exp_options['option_type'] == 'call']

        # Calculate skew metrics
        atm_strike = exp_options[abs(exp_options['moneyness'] - 1.0) < 0.02]['strike'].mean()

        # 25-delta skew (typical measure)
        put_25d = puts[abs(puts['delta'] + 0.25) < 0.05]['implied_vol'].mean() if len(puts) > 0 else np.nan
        call_25d = calls[abs(calls['delta'] - 0.25) < 0.05]['implied_vol'].mean() if len(calls) > 0 else np.nan

        skew = put_25d - call_25d if not np.isnan(put_25d) and not np.isnan(call_25d) else np.nan

        return {
            'atm_strike': atm_strike,
            'put_25d_iv': put_25d,
            'call_25d_iv': call_25d,
            'skew_25d': skew,
            'skew_direction': 'put_premium' if skew > 0 else 'call_premium' if skew < 0 else 'neutral'
        }


class VarianceSwapReplicator:
    """
    Variance Swap Replication using options.

    Replicates variance swap exposure using a portfolio of options
    weighted by 1/K^2.

    ★ Insight ─────────────────────────────────────
    Variance Swap Key Concepts:
    - Var swap pays realized variance - strike variance
    - Can be replicated with options portfolio weighted by 1/K²
    - Pure volatility exposure without path dependency
    - Used to trade realized vs implied variance
    ─────────────────────────────────────────────────
    """

    def __init__(
        self,
        spot: float,
        options_chain: pd.DataFrame,
        risk_free_rate: float = 0.05,
        time_to_maturity: float = 1/12  # Monthly
    ):
        """
        Initialize Variance Swap Replicator.

        Args:
            spot: Current spot price
            options_chain: Options chain DataFrame
            risk_free_rate: Risk-free rate
            time_to_maturity: Time to maturity in years
        """
        self.spot = spot
        self.options_chain = options_chain
        self.r = risk_free_rate
        self.T = time_to_maturity

    def calculate_fair_variance(self) -> float:
        """
        Calculate fair variance using options prices.

        Returns:
            Fair variance (annualized)
        """
        forward = self.spot * np.exp(self.r * self.T)

        # Separate puts and calls around forward
        puts = self.options_chain[
            (self.options_chain['option_type'] == 'put') &
            (self.options_chain['strike'] < forward)
        ].sort_values('strike')

        calls = self.options_chain[
            (self.options_chain['option_type'] == 'call') &
            (self.options_chain['strike'] >= forward)
        ].sort_values('strike')

        # Weight by 1/K^2
        variance = 0

        for _, opt in puts.iterrows():
            K = opt['strike']
            price = opt['price']
            dK = 1.0  # Assume unit strike spacing, adjust if needed
            variance += (2 / self.T) * (price / K ** 2) * dK

        for _, opt in calls.iterrows():
            K = opt['strike']
            price = opt['price']
            dK = 1.0
            variance += (2 / self.T) * (price / K ** 2) * dK

        return variance

    def calculate_strike(self) -> float:
        """
        Calculate variance swap strike.

        Returns:
            Variance swap strike (vol terms)
        """
        fair_var = self.calculate_fair_variance()
        return np.sqrt(fair_var)

    def calculate_replicating_portfolio(self) -> pd.DataFrame:
        """
        Calculate replicating portfolio weights.

        Returns:
            DataFrame with options and weights
        """
        forward = self.spot * np.exp(self.r * self.T)

        portfolio = self.options_chain.copy()
        portfolio['weight'] = 2 / (self.T * portfolio['strike'] ** 2)

        # Normalize
        portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum()

        return portfolio[['strike', 'option_type', 'price', 'weight']]
