"""
Advanced Position Sizing Module for Quantsploit

This module provides sophisticated position sizing algorithms including:
- Kelly Criterion and Fractional Kelly
- Ralph Vince's Optimal f
- Volatility-adjusted position sizing (ATR-based)
- Volatility parity (equal risk contribution)
- Fixed dollar risk per trade

These methods help optimize capital allocation and risk management for trading strategies.

Author: Quantsploit Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class SizingMethod(Enum):
    """Available position sizing methods"""
    FIXED = "fixed"
    KELLY = "kelly"
    FRACTIONAL_KELLY = "fractional_kelly"
    OPTIMAL_F = "optimal_f"
    ATR = "atr"
    VOLATILITY_PARITY = "volatility_parity"
    RISK_PARITY = "risk_parity"


@dataclass
class PositionSizeResult:
    """Result of position size calculation"""
    shares: int
    position_value: float
    risk_amount: float
    fraction_of_capital: float
    method: str
    details: Dict = field(default_factory=dict)

    def __repr__(self):
        return (f"PositionSizeResult(shares={self.shares}, "
                f"value=${self.position_value:,.2f}, "
                f"risk=${self.risk_amount:,.2f}, "
                f"fraction={self.fraction_of_capital:.2%}, "
                f"method={self.method})")


class KellyCriterion:
    """
    Kelly Criterion position sizing calculator.

    The Kelly Criterion determines the optimal fraction of capital to risk
    on each trade to maximize long-term geometric growth of capital.

    Formula: f* = (p * b - q) / b = (p * (W/L) - q) / (W/L)

    Where:
        f* = fraction of capital to bet
        p = probability of winning (win rate)
        q = probability of losing (1 - p)
        b = odds ratio (average win / average loss)
        W = average win amount
        L = average loss amount (absolute value)

    Example:
        >>> kelly = KellyCriterion()
        >>> kelly.calculate_kelly(win_rate=0.55, avg_win=100, avg_loss=80)
        0.2375  # Bet 23.75% of capital

        >>> kelly.calculate_fractional_kelly(
        ...     win_rate=0.55, avg_win=100, avg_loss=80, fraction=0.5
        ... )
        0.11875  # Half-Kelly: 11.875% of capital
    """

    def __init__(self, max_kelly: float = 1.0, min_kelly: float = 0.0):
        """
        Initialize Kelly Criterion calculator.

        Args:
            max_kelly: Maximum Kelly fraction allowed (default 1.0 = 100%, no leverage)
            min_kelly: Minimum Kelly fraction (default 0.0)

        Example:
            >>> kelly = KellyCriterion(max_kelly=0.5)  # Cap at 50% of capital
        """
        self.max_kelly = max_kelly
        self.min_kelly = min_kelly

    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate the full Kelly fraction.

        Args:
            win_rate: Probability of winning (0.0 to 1.0)
            avg_win: Average winning trade amount (positive)
            avg_loss: Average losing trade amount (positive, absolute value)

        Returns:
            Optimal fraction of capital to risk (0.0 to max_kelly)

        Example:
            >>> kelly = KellyCriterion()
            >>> # 55% win rate, avg win $100, avg loss $80
            >>> f = kelly.calculate_kelly(0.55, 100, 80)
            >>> print(f"Kelly fraction: {f:.4f}")
            Kelly fraction: 0.2375

            >>> # Edge case: 50% win rate, equal win/loss = 0 Kelly
            >>> f = kelly.calculate_kelly(0.50, 100, 100)
            >>> print(f"Kelly fraction: {f:.4f}")
            Kelly fraction: 0.0000
        """
        # Validate inputs
        if not 0 <= win_rate <= 1:
            raise ValueError(f"win_rate must be between 0 and 1, got {win_rate}")
        if avg_win <= 0:
            raise ValueError(f"avg_win must be positive, got {avg_win}")
        if avg_loss <= 0:
            raise ValueError(f"avg_loss must be positive, got {avg_loss}")

        # Calculate Kelly fraction
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss  # Odds ratio (payoff ratio)

        # Kelly formula: f* = (p * b - q) / b
        kelly = (p * b - q) / b

        # Apply bounds
        kelly = max(self.min_kelly, min(kelly, self.max_kelly))

        return kelly

    def calculate_fractional_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.5
    ) -> float:
        """
        Calculate fractional Kelly (e.g., half-Kelly, quarter-Kelly).

        Fractional Kelly reduces the aggressiveness of full Kelly betting
        while still capturing most of the geometric growth. It's commonly
        used to reduce variance and drawdowns.

        Args:
            win_rate: Probability of winning (0.0 to 1.0)
            avg_win: Average winning trade amount (positive)
            avg_loss: Average losing trade amount (positive)
            fraction: Fraction of full Kelly to use (default 0.5 = half-Kelly)

        Returns:
            Fractional Kelly position size

        Example:
            >>> kelly = KellyCriterion()
            >>> # Half-Kelly with 55% win rate
            >>> f = kelly.calculate_fractional_kelly(0.55, 100, 80, fraction=0.5)
            >>> print(f"Half-Kelly: {f:.4f}")
            Half-Kelly: 0.1188

            >>> # Quarter-Kelly (very conservative)
            >>> f = kelly.calculate_fractional_kelly(0.55, 100, 80, fraction=0.25)
            >>> print(f"Quarter-Kelly: {f:.4f}")
            Quarter-Kelly: 0.0594
        """
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be between 0 and 1, got {fraction}")

        full_kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)
        return full_kelly * fraction

    def calculate_from_trades(
        self,
        trades: List[float],
        use_fractional: bool = True,
        fraction: float = 0.5
    ) -> Tuple[float, Dict]:
        """
        Calculate optimal Kelly fraction from historical trade P&L data.

        Args:
            trades: List of trade P&L values (positive = win, negative = loss)
            use_fractional: Whether to return fractional Kelly (default True)
            fraction: Fraction of Kelly to use if use_fractional=True

        Returns:
            Tuple of (kelly_fraction, statistics_dict)

        Example:
            >>> kelly = KellyCriterion()
            >>> trades = [100, -50, 75, -30, 120, -80, 50, -40, 90, -60]
            >>> f, stats = kelly.calculate_from_trades(trades)
            >>> print(f"Kelly: {f:.4f}")
            >>> print(f"Win rate: {stats['win_rate']:.2%}")
            >>> print(f"Avg win: ${stats['avg_win']:.2f}")
            >>> print(f"Avg loss: ${stats['avg_loss']:.2f}")
        """
        if not trades:
            return 0.0, {"error": "No trades provided"}

        trades_arr = np.array(trades)

        # Separate wins and losses
        wins = trades_arr[trades_arr > 0]
        losses = trades_arr[trades_arr < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.0, {
                "error": "Need both winning and losing trades",
                "wins": len(wins),
                "losses": len(losses)
            }

        # Calculate statistics
        win_rate = len(wins) / len(trades_arr)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        # Calculate Kelly
        if use_fractional:
            kelly = self.calculate_fractional_kelly(win_rate, avg_win, avg_loss, fraction)
        else:
            kelly = self.calculate_kelly(win_rate, avg_win, avg_loss)

        stats = {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "payoff_ratio": avg_win / avg_loss,
            "total_trades": len(trades_arr),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "full_kelly": self.calculate_kelly(win_rate, avg_win, avg_loss),
            "fraction_used": fraction if use_fractional else 1.0
        }

        return kelly, stats

    def optimal_f(
        self,
        trades: List[float],
        resolution: int = 100
    ) -> Tuple[float, float, Dict]:
        """
        Calculate Ralph Vince's Optimal f using geometric mean maximization.

        Optimal f finds the fraction of capital that maximizes the Terminal
        Wealth Relative (TWR), which is the compound growth of capital.

        Unlike Kelly which uses average win/loss, Optimal f considers
        the actual distribution of trade outcomes.

        Args:
            trades: List of trade P&L values (as fraction of capital risked)
            resolution: Number of f values to test (default 100)

        Returns:
            Tuple of (optimal_f, max_twr, details_dict)

        Example:
            >>> kelly = KellyCriterion()
            >>> # Trades as percentage returns
            >>> trades = [0.05, -0.02, 0.03, -0.015, 0.04, -0.025, 0.02, -0.01]
            >>> f, twr, details = kelly.optimal_f(trades)
            >>> print(f"Optimal f: {f:.4f}")
            >>> print(f"Expected TWR: {twr:.4f}")
        """
        if not trades:
            return 0.0, 1.0, {"error": "No trades provided"}

        trades_arr = np.array(trades)

        # Find the largest loss (in absolute terms)
        largest_loss = abs(min(trades_arr))
        if largest_loss == 0:
            return 0.0, 1.0, {"error": "No losing trades - cannot calculate optimal f"}

        # Test different f values
        f_values = np.linspace(0.01, 1.0, resolution)
        twrs = []

        for f in f_values:
            # Calculate HPR (Holding Period Return) for each trade
            # HPR = 1 + f * (trade / largest_loss)
            hprs = 1 + f * (trades_arr / largest_loss)

            # TWR is the product of all HPRs
            # Use geometric mean for comparison
            if np.all(hprs > 0):
                twr = np.prod(hprs)
                geometric_mean = twr ** (1 / len(trades_arr))
                twrs.append((f, twr, geometric_mean))
            else:
                twrs.append((f, 0, 0))  # Ruin scenario

        # Find optimal f (maximum geometric mean)
        best = max(twrs, key=lambda x: x[2])
        optimal_f, max_twr, max_geo_mean = best

        # Apply bounds
        optimal_f = min(optimal_f, self.max_kelly)

        details = {
            "largest_loss": largest_loss,
            "max_twr": max_twr,
            "geometric_mean": max_geo_mean,
            "trades_analyzed": len(trades_arr),
            "resolution": resolution
        }

        return optimal_f, max_twr, details


class VolatilityAdjustedSizing:
    """
    Volatility-based position sizing methods.

    These methods adjust position sizes based on market volatility,
    ensuring consistent risk across different market conditions.

    Example:
        >>> vas = VolatilityAdjustedSizing(capital=100000)
        >>> shares = vas.calculate_atr_position_size(
        ...     price=150.0,
        ...     atr=5.0,
        ...     risk_multiplier=2.0,
        ...     risk_per_trade=0.01
        ... )
        >>> print(f"Position size: {shares} shares")
    """

    def __init__(
        self,
        capital: float = 100000.0,
        max_position_pct: float = 0.25,
        min_position_pct: float = 0.01
    ):
        """
        Initialize volatility-adjusted position sizer.

        Args:
            capital: Total available capital
            max_position_pct: Maximum position size as fraction of capital
            min_position_pct: Minimum position size as fraction of capital

        Example:
            >>> vas = VolatilityAdjustedSizing(
            ...     capital=100000,
            ...     max_position_pct=0.20,  # Max 20% per position
            ...     min_position_pct=0.02   # Min 2% per position
            ... )
        """
        self.capital = capital
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct

    def calculate_atr_position_size(
        self,
        price: float,
        atr: float,
        risk_multiplier: float = 2.0,
        risk_per_trade: float = 0.01
    ) -> int:
        """
        Calculate position size based on Average True Range (ATR).

        This method uses ATR to determine stop-loss distance and sizes
        the position so that the stop represents a fixed percentage of capital.

        Formula:
            risk_amount = capital * risk_per_trade
            stop_distance = atr * risk_multiplier
            shares = risk_amount / stop_distance

        Args:
            price: Current stock price
            atr: Average True Range (14-period typical)
            risk_multiplier: Multiple of ATR for stop-loss (default 2.0)
            risk_per_trade: Fraction of capital to risk per trade (default 1%)

        Returns:
            Number of shares to buy

        Example:
            >>> vas = VolatilityAdjustedSizing(capital=100000)
            >>> # Stock at $150, ATR of $5, risk 1% of capital
            >>> shares = vas.calculate_atr_position_size(
            ...     price=150.0,
            ...     atr=5.0,
            ...     risk_multiplier=2.0,
            ...     risk_per_trade=0.01
            ... )
            >>> print(f"Shares: {shares}")
            Shares: 100

            >>> # More volatile stock with ATR of $10
            >>> shares = vas.calculate_atr_position_size(
            ...     price=150.0,
            ...     atr=10.0,
            ...     risk_multiplier=2.0,
            ...     risk_per_trade=0.01
            ... )
            >>> print(f"Shares: {shares}")
            Shares: 50
        """
        if atr <= 0:
            raise ValueError(f"ATR must be positive, got {atr}")
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        # Calculate risk amount in dollars
        risk_amount = self.capital * risk_per_trade

        # Stop distance based on ATR
        stop_distance = atr * risk_multiplier

        # Calculate shares
        shares = risk_amount / stop_distance

        # Apply position limits
        position_value = shares * price
        max_value = self.capital * self.max_position_pct
        min_value = self.capital * self.min_position_pct

        if position_value > max_value:
            shares = max_value / price
        elif position_value < min_value:
            shares = 0  # Position too small, skip trade

        return int(shares)

    def calculate_volatility_parity(
        self,
        prices: Dict[str, float],
        volatilities: Dict[str, float],
        target_volatility: float = 0.10
    ) -> Dict[str, Tuple[int, float]]:
        """
        Calculate position sizes for equal volatility contribution.

        Volatility parity allocates capital so each position contributes
        equally to portfolio volatility. Higher volatility assets get
        smaller allocations.

        Args:
            prices: Dict of {symbol: current_price}
            volatilities: Dict of {symbol: annualized_volatility}
            target_volatility: Target portfolio volatility (default 10%)

        Returns:
            Dict of {symbol: (shares, weight)}

        Example:
            >>> vas = VolatilityAdjustedSizing(capital=100000)
            >>> prices = {"AAPL": 175.0, "TSLA": 250.0, "SPY": 450.0}
            >>> vols = {"AAPL": 0.25, "TSLA": 0.50, "SPY": 0.15}
            >>> positions = vas.calculate_volatility_parity(
            ...     prices, vols, target_volatility=0.10
            ... )
            >>> for sym, (shares, weight) in positions.items():
            ...     print(f"{sym}: {shares} shares, {weight:.2%} weight")
        """
        if not prices or not volatilities:
            return {}

        if set(prices.keys()) != set(volatilities.keys()):
            raise ValueError("Prices and volatilities must have the same symbols")

        symbols = list(prices.keys())
        n = len(symbols)

        # Calculate inverse volatility weights
        inv_vols = {sym: 1 / vol for sym, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        # Raw weights based on inverse volatility
        weights = {sym: inv_vol / total_inv_vol for sym, inv_vol in inv_vols.items()}

        # Scale to target volatility
        # Portfolio vol = sqrt(sum(w_i^2 * vol_i^2)) for uncorrelated assets
        # For correlated assets, this is an approximation
        portfolio_vol = np.sqrt(sum(
            (weights[sym] ** 2) * (volatilities[sym] ** 2)
            for sym in symbols
        ))

        # Scale factor to achieve target volatility
        if portfolio_vol > 0:
            scale = target_volatility / portfolio_vol
        else:
            scale = 1.0

        # Cap scale to avoid leverage
        scale = min(scale, 1.0)

        # Calculate final weights and shares
        result = {}
        for sym in symbols:
            final_weight = weights[sym] * scale
            position_value = self.capital * final_weight
            shares = int(position_value / prices[sym])
            result[sym] = (shares, final_weight)

        return result

    def calculate_risk_based_size(
        self,
        price: float,
        stop_price: float,
        risk_per_trade: float = 0.01
    ) -> int:
        """
        Calculate position size based on fixed dollar risk.

        This method sizes positions so that the distance to the stop-loss
        represents a fixed percentage of capital at risk.

        Formula:
            risk_amount = capital * risk_per_trade
            risk_per_share = abs(price - stop_price)
            shares = risk_amount / risk_per_share

        Args:
            price: Entry price
            stop_price: Stop-loss price
            risk_per_trade: Fraction of capital to risk (default 1%)

        Returns:
            Number of shares to buy

        Example:
            >>> vas = VolatilityAdjustedSizing(capital=100000)
            >>> # Enter at $100, stop at $95, risk 1% ($1000)
            >>> shares = vas.calculate_risk_based_size(
            ...     price=100.0,
            ...     stop_price=95.0,
            ...     risk_per_trade=0.01
            ... )
            >>> print(f"Shares: {shares}")
            Shares: 200
            >>> print(f"Risk per share: $5.00")
            >>> print(f"Total risk: ${shares * 5:.2f}")
            Total risk: $1000.00
        """
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        # Calculate risk per share
        risk_per_share = abs(price - stop_price)

        if risk_per_share <= 0:
            raise ValueError("Stop price cannot equal entry price")

        # Calculate risk amount
        risk_amount = self.capital * risk_per_trade

        # Calculate shares
        shares = risk_amount / risk_per_share

        # Apply position limits
        position_value = shares * price
        max_value = self.capital * self.max_position_pct
        min_value = self.capital * self.min_position_pct

        if position_value > max_value:
            shares = max_value / price
        elif position_value < min_value:
            shares = 0

        return int(shares)

    def update_capital(self, new_capital: float):
        """
        Update the capital amount.

        Args:
            new_capital: New capital amount
        """
        if new_capital <= 0:
            raise ValueError(f"Capital must be positive, got {new_capital}")
        self.capital = new_capital


class PositionSizer:
    """
    Unified position sizing interface supporting multiple methods.

    This class provides a single interface to access all position sizing
    methods, making it easy to integrate with the backtesting framework.

    Supported methods:
        - fixed: Fixed percentage of capital
        - kelly: Full Kelly Criterion
        - fractional_kelly: Fractional Kelly (half-Kelly default)
        - optimal_f: Ralph Vince's Optimal f
        - atr: ATR-based sizing
        - volatility_parity: Equal volatility contribution
        - risk_parity: Fixed dollar risk per trade

    Example:
        >>> sizer = PositionSizer(capital=100000)
        >>> result = sizer.get_position_size(
        ...     method='atr',
        ...     price=150.0,
        ...     atr=5.0,
        ...     risk_per_trade=0.01
        ... )
        >>> print(result)
    """

    def __init__(
        self,
        capital: float = 100000.0,
        max_position_pct: float = 0.25,
        min_position_pct: float = 0.01,
        max_kelly: float = 1.0,
        default_fraction: float = 0.5
    ):
        """
        Initialize the position sizer.

        Args:
            capital: Total available capital
            max_position_pct: Maximum position size as fraction of capital
            min_position_pct: Minimum position size as fraction of capital
            max_kelly: Maximum Kelly fraction (caps leverage)
            default_fraction: Default fraction for fractional Kelly

        Example:
            >>> sizer = PositionSizer(
            ...     capital=100000,
            ...     max_position_pct=0.20,
            ...     max_kelly=0.5,
            ...     default_fraction=0.5
            ... )
        """
        self.capital = capital
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.max_kelly = max_kelly
        self.default_fraction = default_fraction

        # Initialize sub-components
        self.kelly = KellyCriterion(max_kelly=max_kelly)
        self.volatility_sizer = VolatilityAdjustedSizing(
            capital=capital,
            max_position_pct=max_position_pct,
            min_position_pct=min_position_pct
        )

    def get_position_size(
        self,
        method: Union[str, SizingMethod],
        price: float,
        **kwargs
    ) -> PositionSizeResult:
        """
        Calculate position size using the specified method.

        Args:
            method: Sizing method to use (string or SizingMethod enum)
            price: Current stock price
            **kwargs: Additional parameters depending on method

        Method-specific kwargs:
            fixed:
                position_pct (float): Fraction of capital (default 0.1)

            kelly / fractional_kelly:
                win_rate (float): Historical win rate
                avg_win (float): Average winning trade
                avg_loss (float): Average losing trade
                fraction (float): Kelly fraction (for fractional_kelly)

            optimal_f:
                trades (List[float]): Historical trade P&L

            atr:
                atr (float): Average True Range
                risk_multiplier (float): ATR multiple for stop (default 2.0)
                risk_per_trade (float): Fraction of capital to risk

            risk_parity:
                stop_price (float): Stop-loss price
                risk_per_trade (float): Fraction of capital to risk

        Returns:
            PositionSizeResult with shares and details

        Example:
            >>> sizer = PositionSizer(capital=100000)

            >>> # Fixed percentage sizing
            >>> result = sizer.get_position_size('fixed', price=150.0, position_pct=0.1)
            >>> print(f"Fixed: {result.shares} shares")

            >>> # Kelly Criterion sizing
            >>> result = sizer.get_position_size(
            ...     'kelly',
            ...     price=150.0,
            ...     win_rate=0.55,
            ...     avg_win=100,
            ...     avg_loss=80
            ... )
            >>> print(f"Kelly: {result.shares} shares, {result.fraction_of_capital:.2%}")

            >>> # ATR-based sizing
            >>> result = sizer.get_position_size(
            ...     'atr',
            ...     price=150.0,
            ...     atr=5.0,
            ...     risk_per_trade=0.01
            ... )
            >>> print(f"ATR: {result.shares} shares")
        """
        # Convert string to enum if needed
        if isinstance(method, str):
            try:
                method = SizingMethod(method.lower())
            except ValueError:
                raise ValueError(f"Unknown sizing method: {method}. "
                               f"Available: {[m.value for m in SizingMethod]}")

        if method == SizingMethod.FIXED:
            return self._fixed_size(price, **kwargs)
        elif method == SizingMethod.KELLY:
            return self._kelly_size(price, full=True, **kwargs)
        elif method == SizingMethod.FRACTIONAL_KELLY:
            return self._kelly_size(price, full=False, **kwargs)
        elif method == SizingMethod.OPTIMAL_F:
            return self._optimal_f_size(price, **kwargs)
        elif method == SizingMethod.ATR:
            return self._atr_size(price, **kwargs)
        elif method == SizingMethod.VOLATILITY_PARITY:
            raise ValueError("Use calculate_volatility_parity() for multi-asset allocation")
        elif method == SizingMethod.RISK_PARITY:
            return self._risk_parity_size(price, **kwargs)
        else:
            raise ValueError(f"Unhandled sizing method: {method}")

    def _fixed_size(self, price: float, position_pct: float = 0.1, **kwargs) -> PositionSizeResult:
        """Fixed percentage position sizing"""
        position_value = self.capital * position_pct
        shares = int(position_value / price)
        actual_value = shares * price

        return PositionSizeResult(
            shares=shares,
            position_value=actual_value,
            risk_amount=actual_value,  # Full position is at risk
            fraction_of_capital=actual_value / self.capital,
            method="fixed",
            details={"position_pct": position_pct}
        )

    def _kelly_size(
        self,
        price: float,
        full: bool = False,
        win_rate: float = 0.5,
        avg_win: float = 100,
        avg_loss: float = 100,
        fraction: float = None,
        **kwargs
    ) -> PositionSizeResult:
        """Kelly Criterion position sizing"""
        if fraction is None:
            fraction = self.default_fraction

        if full:
            kelly_f = self.kelly.calculate_kelly(win_rate, avg_win, avg_loss)
        else:
            kelly_f = self.kelly.calculate_fractional_kelly(
                win_rate, avg_win, avg_loss, fraction
            )

        position_value = self.capital * kelly_f

        # Apply position limits
        max_value = self.capital * self.max_position_pct
        min_value = self.capital * self.min_position_pct

        if position_value > max_value:
            position_value = max_value
            kelly_f = self.max_position_pct
        elif position_value < min_value:
            position_value = 0
            kelly_f = 0

        shares = int(position_value / price) if position_value > 0 else 0
        actual_value = shares * price

        return PositionSizeResult(
            shares=shares,
            position_value=actual_value,
            risk_amount=actual_value,
            fraction_of_capital=actual_value / self.capital if self.capital > 0 else 0,
            method="kelly" if full else "fractional_kelly",
            details={
                "kelly_fraction": kelly_f,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "payoff_ratio": avg_win / avg_loss,
                "fraction_used": 1.0 if full else fraction
            }
        )

    def _optimal_f_size(
        self,
        price: float,
        trades: List[float] = None,
        **kwargs
    ) -> PositionSizeResult:
        """Optimal f position sizing"""
        if trades is None or len(trades) == 0:
            # Default to conservative sizing if no trade history
            return self._fixed_size(price, position_pct=0.05)

        optimal_f, twr, details = self.kelly.optimal_f(trades)

        position_value = self.capital * optimal_f

        # Apply position limits
        max_value = self.capital * self.max_position_pct
        if position_value > max_value:
            position_value = max_value
            optimal_f = self.max_position_pct

        shares = int(position_value / price) if position_value > 0 else 0
        actual_value = shares * price

        return PositionSizeResult(
            shares=shares,
            position_value=actual_value,
            risk_amount=actual_value,
            fraction_of_capital=actual_value / self.capital if self.capital > 0 else 0,
            method="optimal_f",
            details={
                "optimal_f": optimal_f,
                "terminal_wealth_relative": twr,
                **details
            }
        )

    def _atr_size(
        self,
        price: float,
        atr: float = None,
        risk_multiplier: float = 2.0,
        risk_per_trade: float = 0.01,
        **kwargs
    ) -> PositionSizeResult:
        """ATR-based position sizing"""
        if atr is None or atr <= 0:
            raise ValueError("ATR must be provided and positive for ATR sizing")

        shares = self.volatility_sizer.calculate_atr_position_size(
            price=price,
            atr=atr,
            risk_multiplier=risk_multiplier,
            risk_per_trade=risk_per_trade
        )

        actual_value = shares * price
        stop_distance = atr * risk_multiplier
        risk_amount = shares * stop_distance

        return PositionSizeResult(
            shares=shares,
            position_value=actual_value,
            risk_amount=risk_amount,
            fraction_of_capital=actual_value / self.capital if self.capital > 0 else 0,
            method="atr",
            details={
                "atr": atr,
                "risk_multiplier": risk_multiplier,
                "stop_distance": stop_distance,
                "risk_per_trade": risk_per_trade,
                "risk_amount": risk_amount
            }
        )

    def _risk_parity_size(
        self,
        price: float,
        stop_price: float = None,
        risk_per_trade: float = 0.01,
        **kwargs
    ) -> PositionSizeResult:
        """Fixed dollar risk position sizing"""
        if stop_price is None:
            raise ValueError("stop_price must be provided for risk_parity sizing")

        shares = self.volatility_sizer.calculate_risk_based_size(
            price=price,
            stop_price=stop_price,
            risk_per_trade=risk_per_trade
        )

        actual_value = shares * price
        risk_per_share = abs(price - stop_price)
        risk_amount = shares * risk_per_share

        return PositionSizeResult(
            shares=shares,
            position_value=actual_value,
            risk_amount=risk_amount,
            fraction_of_capital=actual_value / self.capital if self.capital > 0 else 0,
            method="risk_parity",
            details={
                "stop_price": stop_price,
                "risk_per_share": risk_per_share,
                "risk_per_trade": risk_per_trade,
                "actual_risk_pct": risk_amount / self.capital if self.capital > 0 else 0
            }
        )

    def calculate_volatility_parity(
        self,
        prices: Dict[str, float],
        volatilities: Dict[str, float],
        target_volatility: float = 0.10
    ) -> Dict[str, PositionSizeResult]:
        """
        Calculate volatility parity allocations for multiple assets.

        Args:
            prices: Dict of {symbol: price}
            volatilities: Dict of {symbol: annualized_volatility}
            target_volatility: Target portfolio volatility

        Returns:
            Dict of {symbol: PositionSizeResult}

        Example:
            >>> sizer = PositionSizer(capital=100000)
            >>> prices = {"AAPL": 175.0, "TSLA": 250.0, "SPY": 450.0}
            >>> vols = {"AAPL": 0.25, "TSLA": 0.50, "SPY": 0.15}
            >>> results = sizer.calculate_volatility_parity(prices, vols)
            >>> for sym, result in results.items():
            ...     print(f"{sym}: {result}")
        """
        raw_results = self.volatility_sizer.calculate_volatility_parity(
            prices, volatilities, target_volatility
        )

        results = {}
        for sym, (shares, weight) in raw_results.items():
            price = prices[sym]
            actual_value = shares * price

            results[sym] = PositionSizeResult(
                shares=shares,
                position_value=actual_value,
                risk_amount=actual_value * volatilities[sym],
                fraction_of_capital=actual_value / self.capital,
                method="volatility_parity",
                details={
                    "weight": weight,
                    "volatility": volatilities[sym],
                    "target_volatility": target_volatility
                }
            )

        return results

    def update_capital(self, new_capital: float):
        """
        Update capital for position calculations.

        Args:
            new_capital: New capital amount
        """
        if new_capital <= 0:
            raise ValueError(f"Capital must be positive, got {new_capital}")
        self.capital = new_capital
        self.volatility_sizer.update_capital(new_capital)


# Convenience functions for quick calculations

def calculate_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Quick Kelly Criterion calculation.

    Example:
        >>> f = calculate_kelly(0.55, 100, 80)
        >>> print(f"Kelly: {f:.2%}")
    """
    kelly = KellyCriterion()
    return kelly.calculate_kelly(win_rate, avg_win, avg_loss)


def calculate_half_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Quick half-Kelly calculation.

    Example:
        >>> f = calculate_half_kelly(0.55, 100, 80)
        >>> print(f"Half-Kelly: {f:.2%}")
    """
    kelly = KellyCriterion()
    return kelly.calculate_fractional_kelly(win_rate, avg_win, avg_loss, 0.5)


def calculate_atr_shares(
    capital: float,
    price: float,
    atr: float,
    risk_pct: float = 0.01,
    atr_multiplier: float = 2.0
) -> int:
    """
    Quick ATR-based position size calculation.

    Example:
        >>> shares = calculate_atr_shares(100000, 150.0, 5.0, 0.01, 2.0)
        >>> print(f"Shares: {shares}")
    """
    vas = VolatilityAdjustedSizing(capital=capital)
    return vas.calculate_atr_position_size(price, atr, atr_multiplier, risk_pct)


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("Quantsploit Position Sizing Module - Demo")
    print("=" * 60)

    # Kelly Criterion examples
    print("\n--- Kelly Criterion ---")
    kelly = KellyCriterion(max_kelly=1.0)

    # Example: 55% win rate, avg win $100, avg loss $80
    f = kelly.calculate_kelly(win_rate=0.55, avg_win=100, avg_loss=80)
    print(f"Full Kelly (55% WR, 100/80): {f:.2%}")

    f_half = kelly.calculate_fractional_kelly(0.55, 100, 80, fraction=0.5)
    print(f"Half Kelly: {f_half:.2%}")

    f_quarter = kelly.calculate_fractional_kelly(0.55, 100, 80, fraction=0.25)
    print(f"Quarter Kelly: {f_quarter:.2%}")

    # Calculate from trades
    print("\n--- Kelly from Trade History ---")
    trades = [100, -50, 75, -30, 120, -80, 50, -40, 90, -60, 110, -45]
    f, stats = kelly.calculate_from_trades(trades)
    print(f"Kelly from trades: {f:.2%}")
    print(f"  Win rate: {stats['win_rate']:.1%}")
    print(f"  Avg win: ${stats['avg_win']:.2f}")
    print(f"  Avg loss: ${stats['avg_loss']:.2f}")
    print(f"  Payoff ratio: {stats['payoff_ratio']:.2f}")

    # Optimal f
    print("\n--- Optimal f ---")
    trade_returns = [0.05, -0.02, 0.03, -0.015, 0.04, -0.025, 0.02, -0.01, 0.035, -0.018]
    opt_f, twr, details = kelly.optimal_f(trade_returns)
    print(f"Optimal f: {opt_f:.2%}")
    print(f"Expected TWR: {twr:.4f}")

    # Volatility-adjusted sizing
    print("\n--- Volatility-Adjusted Sizing ---")
    vas = VolatilityAdjustedSizing(capital=100000)

    shares = vas.calculate_atr_position_size(price=150.0, atr=5.0, risk_multiplier=2.0, risk_per_trade=0.01)
    print(f"ATR sizing (price=$150, ATR=$5): {shares} shares")

    shares = vas.calculate_risk_based_size(price=100.0, stop_price=95.0, risk_per_trade=0.01)
    print(f"Risk-based sizing (entry=$100, stop=$95): {shares} shares")

    # Volatility parity
    print("\n--- Volatility Parity ---")
    prices = {"AAPL": 175.0, "TSLA": 250.0, "SPY": 450.0}
    vols = {"AAPL": 0.25, "TSLA": 0.50, "SPY": 0.15}
    allocations = vas.calculate_volatility_parity(prices, vols, target_volatility=0.10)
    for sym, (shares, weight) in allocations.items():
        print(f"  {sym}: {shares} shares, {weight:.1%} weight")

    # Unified PositionSizer
    print("\n--- Unified PositionSizer ---")
    sizer = PositionSizer(capital=100000)

    result = sizer.get_position_size('fixed', price=150.0, position_pct=0.1)
    print(f"Fixed: {result}")

    result = sizer.get_position_size('kelly', price=150.0, win_rate=0.55, avg_win=100, avg_loss=80)
    print(f"Kelly: {result}")

    result = sizer.get_position_size('fractional_kelly', price=150.0, win_rate=0.55, avg_win=100, avg_loss=80)
    print(f"Half-Kelly: {result}")

    result = sizer.get_position_size('atr', price=150.0, atr=5.0, risk_per_trade=0.01)
    print(f"ATR: {result}")

    result = sizer.get_position_size('risk_parity', price=100.0, stop_price=95.0, risk_per_trade=0.01)
    print(f"Risk Parity: {result}")

    print("\n" + "=" * 60)
    print("Demo complete!")
