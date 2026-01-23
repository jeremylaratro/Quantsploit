"""
Transaction Cost Model for Quantsploit

This module provides realistic transaction cost modeling including:
- Tiered commission structures
- Price and volatility-dependent slippage
- Non-linear market impact costs (Almgren-Chriss style)
- Bid-ask spread costs
- Pre-configured cost profiles for different trader types

Market Impact Model Reference:
    Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
    Journal of Risk, 3(2), 5-39.

The market impact follows: impact_bps = eta * (participation_rate ** alpha)
where:
    - participation_rate = shares / average_daily_volume
    - eta = base impact coefficient (varies by liquidity tier)
    - alpha = impact exponent (typically 0.5-0.8)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from quantsploit.utils.backtesting import (
    Backtester, BacktestConfig, BacktestResults,
    Trade, Position, PositionSide
)


class LiquidityTier(Enum):
    """Liquidity tier classification for market impact estimation"""
    LARGE_CAP = "large_cap"      # >$10B market cap, high liquidity
    MID_CAP = "mid_cap"          # $2B-$10B market cap
    SMALL_CAP = "small_cap"      # $300M-$2B market cap
    MICRO_CAP = "micro_cap"      # <$300M market cap, low liquidity


class CostProfile(Enum):
    """Pre-defined cost profiles for different trader types"""
    RETAIL = "retail"            # Higher commissions, standard slippage
    INSTITUTIONAL = "institutional"  # Lower commissions, market impact focus
    HFT = "hft"                  # Minimal commission, tight spreads


@dataclass
class CommissionTier:
    """
    Represents a commission tier based on monthly trading volume

    Attributes:
        min_volume: Minimum monthly volume for this tier (in dollars)
        rate: Commission rate as a percentage (0.001 = 0.1%)
        per_share: Optional per-share commission (e.g., $0.005/share)
        minimum: Minimum commission per trade
    """
    min_volume: float
    rate: float
    per_share: float = 0.0
    minimum: float = 0.0


@dataclass
class MarketImpactParams:
    """
    Parameters for market impact calculation

    Based on the square-root model: impact = eta * sqrt(Q/V) * sigma
    where Q = order size, V = ADV, sigma = volatility

    Simplified as: impact_bps = eta * (participation_rate ** alpha)

    Attributes:
        eta: Base impact coefficient (basis points)
        alpha: Impact exponent (typically 0.5-0.8)
        temporary_factor: Fraction of impact that is temporary (reverts)
        permanent_factor: Fraction of impact that is permanent
    """
    eta: float = 10.0           # Base impact in bps
    alpha: float = 0.6          # Power law exponent
    temporary_factor: float = 0.7   # 70% temporary impact
    permanent_factor: float = 0.3   # 30% permanent impact


@dataclass
class TransactionCostBreakdown:
    """
    Detailed breakdown of transaction costs for a trade

    All costs are in absolute dollar terms.
    """
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    spread_cost: float = 0.0
    total_cost: float = 0.0

    # Cost as percentage of trade value
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    spread_cost_bps: float = 0.0
    total_bps: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for reporting"""
        return {
            'commission': self.commission,
            'slippage': self.slippage,
            'market_impact': self.market_impact,
            'spread_cost': self.spread_cost,
            'total_cost': self.total_cost,
            'commission_bps': self.commission_bps,
            'slippage_bps': self.slippage_bps,
            'market_impact_bps': self.market_impact_bps,
            'spread_cost_bps': self.spread_cost_bps,
            'total_bps': self.total_bps
        }


class TransactionCostModel:
    """
    Comprehensive transaction cost model for realistic backtesting

    Components:
        1. Commission: Tiered structure based on volume
        2. Slippage: Price-dependent execution slippage
        3. Market Impact: Non-linear cost proportional to (volume/ADV)^alpha
        4. Bid-Ask Spread: Cost of crossing the spread

    Example usage:
        >>> model = TransactionCostModel(profile=CostProfile.RETAIL)
        >>> cost = model.total_transaction_cost(
        ...     price=100.0,
        ...     shares=1000,
        ...     avg_daily_volume=1000000,
        ...     volatility=0.02,
        ...     spread_bps=5.0
        ... )
        >>> print(f"Total cost: ${cost.total_cost:.2f}")
    """

    # Default commission tiers (retail)
    DEFAULT_COMMISSION_TIERS = [
        CommissionTier(min_volume=0, rate=0.001, minimum=1.0),          # 0.1%, min $1
        CommissionTier(min_volume=10000, rate=0.0008, minimum=1.0),     # 0.08%
        CommissionTier(min_volume=100000, rate=0.0005, minimum=1.0),    # 0.05%
        CommissionTier(min_volume=1000000, rate=0.0003, minimum=1.0),   # 0.03%
    ]

    # Market impact parameters by liquidity tier
    LIQUIDITY_IMPACT_PARAMS = {
        LiquidityTier.LARGE_CAP: MarketImpactParams(eta=5.0, alpha=0.5),
        LiquidityTier.MID_CAP: MarketImpactParams(eta=10.0, alpha=0.55),
        LiquidityTier.SMALL_CAP: MarketImpactParams(eta=20.0, alpha=0.6),
        LiquidityTier.MICRO_CAP: MarketImpactParams(eta=40.0, alpha=0.65),
    }

    # Default spread by liquidity tier (in basis points)
    DEFAULT_SPREADS_BPS = {
        LiquidityTier.LARGE_CAP: 1.0,    # 1 bps
        LiquidityTier.MID_CAP: 3.0,      # 3 bps
        LiquidityTier.SMALL_CAP: 10.0,   # 10 bps
        LiquidityTier.MICRO_CAP: 25.0,   # 25 bps
    }

    # Pre-configured cost profiles
    COST_PROFILES = {
        CostProfile.RETAIL: {
            'commission_tiers': [
                CommissionTier(min_volume=0, rate=0.001, minimum=4.95),
                CommissionTier(min_volume=50000, rate=0.0007, minimum=4.95),
                CommissionTier(min_volume=500000, rate=0.0004, minimum=4.95),
            ],
            'slippage_base_bps': 5.0,       # 5 bps base slippage
            'slippage_volatility_mult': 2.0,  # Multiplier for volatility
            'spread_multiplier': 1.0,         # Pay full spread
        },
        CostProfile.INSTITUTIONAL: {
            'commission_tiers': [
                CommissionTier(min_volume=0, rate=0.0003, per_share=0.003, minimum=1.0),
                CommissionTier(min_volume=1000000, rate=0.0002, per_share=0.002, minimum=1.0),
                CommissionTier(min_volume=10000000, rate=0.0001, per_share=0.001, minimum=1.0),
            ],
            'slippage_base_bps': 2.0,
            'slippage_volatility_mult': 1.5,
            'spread_multiplier': 0.6,  # Better execution, pay ~60% of spread
        },
        CostProfile.HFT: {
            'commission_tiers': [
                CommissionTier(min_volume=0, rate=0.00005, per_share=0.0005, minimum=0.0),
            ],
            'slippage_base_bps': 0.5,
            'slippage_volatility_mult': 1.0,
            'spread_multiplier': 0.3,  # Maker rebates, tight spreads
        },
    }

    def __init__(
        self,
        profile: Optional[CostProfile] = None,
        commission_tiers: Optional[List[CommissionTier]] = None,
        slippage_base_bps: float = 5.0,
        slippage_volatility_mult: float = 2.0,
        spread_multiplier: float = 1.0,
        liquidity_tier: LiquidityTier = LiquidityTier.MID_CAP,
        impact_params: Optional[MarketImpactParams] = None,
        monthly_volume: float = 0.0
    ):
        """
        Initialize the transaction cost model

        Args:
            profile: Pre-configured cost profile (overrides other params if set)
            commission_tiers: Custom commission tier structure
            slippage_base_bps: Base slippage in basis points
            slippage_volatility_mult: Volatility multiplier for slippage
            spread_multiplier: Fraction of spread paid (1.0 = full spread)
            liquidity_tier: Default liquidity classification
            impact_params: Custom market impact parameters
            monthly_volume: Monthly trading volume for commission tier calculation
        """
        # Apply profile settings if provided
        if profile is not None:
            profile_config = self.COST_PROFILES[profile]
            self.commission_tiers = profile_config['commission_tiers']
            self.slippage_base_bps = profile_config['slippage_base_bps']
            self.slippage_volatility_mult = profile_config['slippage_volatility_mult']
            self.spread_multiplier = profile_config['spread_multiplier']
        else:
            self.commission_tiers = commission_tiers or self.DEFAULT_COMMISSION_TIERS
            self.slippage_base_bps = slippage_base_bps
            self.slippage_volatility_mult = slippage_volatility_mult
            self.spread_multiplier = spread_multiplier

        self.liquidity_tier = liquidity_tier
        self.impact_params = impact_params or self.LIQUIDITY_IMPACT_PARAMS[liquidity_tier]
        self.monthly_volume = monthly_volume

        # Sort tiers by min_volume (descending) for lookup
        self.commission_tiers = sorted(
            self.commission_tiers,
            key=lambda x: x.min_volume,
            reverse=True
        )

    def calculate_commission(
        self,
        price: float,
        shares: int,
        monthly_volume: Optional[float] = None
    ) -> float:
        """
        Calculate commission based on tiered structure

        The commission is calculated as:
            commission = max(rate * trade_value, per_share * shares, minimum)

        Args:
            price: Execution price per share
            shares: Number of shares
            monthly_volume: Monthly trading volume for tier determination

        Returns:
            Commission in dollars
        """
        trade_value = price * shares
        volume = monthly_volume if monthly_volume is not None else self.monthly_volume

        # Find applicable tier
        tier = self.commission_tiers[-1]  # Default to lowest tier
        for t in self.commission_tiers:
            if volume >= t.min_volume:
                tier = t
                break

        # Calculate commission components
        rate_commission = trade_value * tier.rate
        share_commission = shares * tier.per_share

        # Use the higher of rate-based or per-share commission
        commission = max(rate_commission, share_commission)

        # Apply minimum
        return max(commission, tier.minimum)

    def calculate_slippage(
        self,
        price: float,
        shares: int,
        volatility: float = 0.02,
        is_market_order: bool = True
    ) -> float:
        """
        Calculate execution slippage

        Slippage is modeled as a function of:
            - Base slippage (execution uncertainty)
            - Volatility (higher volatility = more slippage)
            - Order type (market orders have more slippage)

        Args:
            price: Execution price per share
            shares: Number of shares
            volatility: Daily volatility (as decimal, e.g., 0.02 = 2%)
            is_market_order: Whether this is a market order

        Returns:
            Slippage cost in dollars
        """
        trade_value = price * shares

        # Base slippage in bps
        slippage_bps = self.slippage_base_bps

        # Volatility adjustment: higher volatility = more slippage
        # Volatility is scaled to daily (0.02 = 2% daily vol is typical)
        volatility_adjustment = volatility * self.slippage_volatility_mult * 100
        slippage_bps += volatility_adjustment

        # Market order penalty (limit orders have less slippage)
        if not is_market_order:
            slippage_bps *= 0.5

        # Convert bps to dollars
        slippage = trade_value * (slippage_bps / 10000)

        return slippage

    def calculate_market_impact(
        self,
        price: float,
        shares: int,
        avg_daily_volume: float,
        volatility: float = 0.02,
        liquidity_tier: Optional[LiquidityTier] = None
    ) -> float:
        """
        Calculate market impact cost using Almgren-Chriss style model

        The model: impact_bps = eta * (participation_rate ** alpha)

        Where:
            - participation_rate = shares / avg_daily_volume
            - eta = base impact coefficient
            - alpha = impact exponent (sub-linear, typically 0.5-0.8)

        Args:
            price: Execution price per share
            shares: Number of shares
            avg_daily_volume: Average daily volume in shares
            volatility: Daily volatility (as decimal)
            liquidity_tier: Liquidity classification for impact params

        Returns:
            Market impact cost in dollars
        """
        if avg_daily_volume <= 0:
            avg_daily_volume = shares * 10  # Assume we're 10% of volume if unknown

        trade_value = price * shares

        # Get impact parameters
        tier = liquidity_tier or self.liquidity_tier
        params = self.LIQUIDITY_IMPACT_PARAMS.get(tier, self.impact_params)

        # Calculate participation rate
        participation_rate = shares / avg_daily_volume

        # Non-linear impact: impact_bps = eta * (participation_rate ** alpha)
        # This captures the square-root rule empirically observed in markets
        impact_bps = params.eta * (participation_rate ** params.alpha)

        # Volatility scaling: higher vol = more impact
        # Normalize to typical 2% daily volatility
        vol_scalar = volatility / 0.02
        impact_bps *= vol_scalar

        # Cap at reasonable maximum (100 bps = 1%)
        impact_bps = min(impact_bps, 100.0)

        # Convert to dollars
        impact = trade_value * (impact_bps / 10000)

        return impact

    def calculate_bid_ask_spread(
        self,
        price: float,
        shares: int,
        spread_bps: Optional[float] = None,
        liquidity_tier: Optional[LiquidityTier] = None
    ) -> float:
        """
        Calculate bid-ask spread crossing cost

        When you buy at the ask or sell at the bid, you pay half the spread.

        Args:
            price: Mid-price per share
            shares: Number of shares
            spread_bps: Bid-ask spread in basis points (if known)
            liquidity_tier: Liquidity tier for default spread estimation

        Returns:
            Spread cost in dollars
        """
        trade_value = price * shares

        # Get spread
        if spread_bps is None:
            tier = liquidity_tier or self.liquidity_tier
            spread_bps = self.DEFAULT_SPREADS_BPS.get(tier, 5.0)

        # Pay half the spread to cross (buy at ask, sell at bid)
        half_spread_bps = spread_bps / 2.0

        # Apply spread multiplier (institutional may get better fills)
        effective_spread_bps = half_spread_bps * self.spread_multiplier

        # Convert to dollars
        spread_cost = trade_value * (effective_spread_bps / 10000)

        return spread_cost

    def total_transaction_cost(
        self,
        price: float,
        shares: int,
        avg_daily_volume: float = 1000000,
        volatility: float = 0.02,
        spread_bps: Optional[float] = None,
        liquidity_tier: Optional[LiquidityTier] = None,
        is_market_order: bool = True,
        monthly_volume: Optional[float] = None
    ) -> TransactionCostBreakdown:
        """
        Calculate total transaction cost with full breakdown

        Args:
            price: Execution price per share
            shares: Number of shares
            avg_daily_volume: Average daily volume in shares
            volatility: Daily volatility (as decimal)
            spread_bps: Bid-ask spread in basis points
            liquidity_tier: Liquidity tier for market impact
            is_market_order: Whether this is a market order
            monthly_volume: Monthly trading volume for commission tiers

        Returns:
            TransactionCostBreakdown with all cost components
        """
        trade_value = price * shares
        tier = liquidity_tier or self.liquidity_tier

        # Calculate each component
        commission = self.calculate_commission(price, shares, monthly_volume)
        slippage = self.calculate_slippage(price, shares, volatility, is_market_order)
        market_impact = self.calculate_market_impact(
            price, shares, avg_daily_volume, volatility, tier
        )
        spread_cost = self.calculate_bid_ask_spread(price, shares, spread_bps, tier)

        # Total
        total = commission + slippage + market_impact + spread_cost

        # Convert to basis points
        if trade_value > 0:
            commission_bps = (commission / trade_value) * 10000
            slippage_bps = (slippage / trade_value) * 10000
            market_impact_bps = (market_impact / trade_value) * 10000
            spread_cost_bps = (spread_cost / trade_value) * 10000
            total_bps = (total / trade_value) * 10000
        else:
            commission_bps = slippage_bps = market_impact_bps = spread_cost_bps = total_bps = 0.0

        return TransactionCostBreakdown(
            commission=commission,
            slippage=slippage,
            market_impact=market_impact,
            spread_cost=spread_cost,
            total_cost=total,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            market_impact_bps=market_impact_bps,
            spread_cost_bps=spread_cost_bps,
            total_bps=total_bps
        )

    def estimate_liquidity_tier(
        self,
        avg_daily_volume: float,
        price: float,
        market_cap: Optional[float] = None
    ) -> LiquidityTier:
        """
        Estimate liquidity tier from market data

        Uses average daily dollar volume or market cap to classify.

        Args:
            avg_daily_volume: Average daily volume in shares
            price: Current price per share
            market_cap: Market capitalization (optional)

        Returns:
            Estimated LiquidityTier
        """
        # Use market cap if available
        if market_cap is not None:
            if market_cap >= 10e9:
                return LiquidityTier.LARGE_CAP
            elif market_cap >= 2e9:
                return LiquidityTier.MID_CAP
            elif market_cap >= 300e6:
                return LiquidityTier.SMALL_CAP
            else:
                return LiquidityTier.MICRO_CAP

        # Otherwise use average daily dollar volume
        adv_dollars = avg_daily_volume * price

        if adv_dollars >= 100e6:      # $100M+ daily volume
            return LiquidityTier.LARGE_CAP
        elif adv_dollars >= 20e6:     # $20M+ daily volume
            return LiquidityTier.MID_CAP
        elif adv_dollars >= 2e6:      # $2M+ daily volume
            return LiquidityTier.SMALL_CAP
        else:
            return LiquidityTier.MICRO_CAP


@dataclass
class CostAwareBacktestConfig(BacktestConfig):
    """
    Extended backtest configuration with advanced cost modeling

    Inherits from BacktestConfig and adds cost model parameters.
    """
    cost_model: Optional[TransactionCostModel] = None
    cost_profile: Optional[CostProfile] = None
    track_cost_breakdown: bool = True
    use_volume_for_impact: bool = True
    default_adv_multiplier: float = 100.0  # Default: trade is 1/100 of ADV


@dataclass
class CostAwareBacktestResults(BacktestResults):
    """
    Extended backtest results with transaction cost analytics
    """
    # Cost breakdown totals
    total_commissions: float = 0.0
    total_slippage: float = 0.0
    total_market_impact: float = 0.0
    total_spread_costs: float = 0.0
    total_transaction_costs: float = 0.0

    # Cost as percentage of traded value
    avg_cost_bps: float = 0.0

    # Per-trade cost breakdown
    cost_breakdown_by_trade: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert results to dictionary including cost breakdown"""
        base_dict = super().to_dict()
        base_dict.update({
            "Total Commissions": f"${self.total_commissions:,.2f}",
            "Total Slippage": f"${self.total_slippage:,.2f}",
            "Total Market Impact": f"${self.total_market_impact:,.2f}",
            "Total Spread Costs": f"${self.total_spread_costs:,.2f}",
            "Total Transaction Costs": f"${self.total_transaction_costs:,.2f}",
            "Average Cost (bps)": f"{self.avg_cost_bps:.2f}",
        })
        return base_dict


class CostAwareBacktester(Backtester):
    """
    Backtester with advanced transaction cost modeling

    Extends the base Backtester to incorporate:
        - Tiered commission structures
        - Price-dependent slippage
        - Market impact (volume/ADV based)
        - Bid-ask spread costs

    Example usage:
        >>> from quantsploit.utils.transaction_costs import (
        ...     CostAwareBacktester, CostAwareBacktestConfig, CostProfile
        ... )
        >>>
        >>> config = CostAwareBacktestConfig(
        ...     initial_capital=100000,
        ...     cost_profile=CostProfile.RETAIL
        ... )
        >>> backtester = CostAwareBacktester(config)
        >>>
        >>> # Run backtest with volume data for market impact
        >>> results = backtester.run_backtest(data, strategy_func, symbol='AAPL')
    """

    def __init__(self, config: CostAwareBacktestConfig = None):
        """
        Initialize the cost-aware backtester

        Args:
            config: CostAwareBacktestConfig with cost model settings
        """
        # Convert to CostAwareBacktestConfig if needed
        if config is None:
            config = CostAwareBacktestConfig()
        elif not isinstance(config, CostAwareBacktestConfig):
            # Convert BacktestConfig to CostAwareBacktestConfig
            config = CostAwareBacktestConfig(
                initial_capital=config.initial_capital,
                commission_pct=config.commission_pct,
                commission_min=config.commission_min,
                slippage_pct=config.slippage_pct,
                position_size=config.position_size,
                max_positions=config.max_positions,
                margin_requirement=config.margin_requirement,
                risk_free_rate=config.risk_free_rate,
                benchmark_symbol=config.benchmark_symbol
            )

        super().__init__(config)
        self.cost_config = config

        # Initialize cost model
        if config.cost_model is not None:
            self.cost_model = config.cost_model
        elif config.cost_profile is not None:
            self.cost_model = TransactionCostModel(profile=config.cost_profile)
        else:
            self.cost_model = TransactionCostModel()

        # Track cost breakdowns
        self.cost_breakdowns: List[TransactionCostBreakdown] = []
        self.current_volume_data: Optional[pd.Series] = None
        self.current_volatility: float = 0.02

    def reset(self):
        """Reset backtester state including cost tracking"""
        super().reset()
        self.cost_breakdowns = []
        self.current_volume_data = None
        self.current_volatility = 0.02

    def calculate_commission(self, price: float, shares: int) -> float:
        """
        Calculate commission using the advanced cost model

        Overrides the base implementation to use tiered commissions.
        """
        return self.cost_model.calculate_commission(price, shares)

    def calculate_slippage(self, price: float, shares: int) -> float:
        """
        Calculate slippage using the advanced cost model

        Overrides the base implementation to include volatility-dependent slippage.
        """
        return self.cost_model.calculate_slippage(
            price, shares,
            volatility=self.current_volatility
        )

    def calculate_market_impact(
        self,
        price: float,
        shares: int,
        avg_daily_volume: Optional[float] = None
    ) -> float:
        """
        Calculate market impact cost

        Args:
            price: Execution price
            shares: Number of shares
            avg_daily_volume: Average daily volume (optional, uses estimate if not provided)

        Returns:
            Market impact cost in dollars
        """
        # Estimate ADV if not provided
        if avg_daily_volume is None:
            trade_value = price * shares
            avg_daily_volume = shares * self.cost_config.default_adv_multiplier

        return self.cost_model.calculate_market_impact(
            price, shares, avg_daily_volume,
            volatility=self.current_volatility
        )

    def calculate_spread_cost(self, price: float, shares: int) -> float:
        """
        Calculate bid-ask spread crossing cost

        Args:
            price: Mid-price
            shares: Number of shares

        Returns:
            Spread cost in dollars
        """
        return self.cost_model.calculate_bid_ask_spread(price, shares)

    def get_total_transaction_cost(
        self,
        price: float,
        shares: int,
        avg_daily_volume: Optional[float] = None
    ) -> TransactionCostBreakdown:
        """
        Get complete transaction cost breakdown

        Args:
            price: Execution price
            shares: Number of shares
            avg_daily_volume: Average daily volume

        Returns:
            TransactionCostBreakdown with all components
        """
        if avg_daily_volume is None:
            avg_daily_volume = shares * self.cost_config.default_adv_multiplier

        return self.cost_model.total_transaction_cost(
            price=price,
            shares=shares,
            avg_daily_volume=avg_daily_volume,
            volatility=self.current_volatility
        )

    def enter_long(
        self,
        symbol: str,
        date: datetime,
        price: float,
        shares: int = None,
        avg_daily_volume: Optional[float] = None
    ) -> bool:
        """
        Enter a long position with advanced cost modeling

        Overrides base implementation to track cost breakdown.

        Args:
            symbol: Stock symbol
            date: Entry date
            price: Entry price
            shares: Number of shares (calculated if not provided)
            avg_daily_volume: Average daily volume for impact calculation

        Returns:
            True if position was opened successfully
        """
        if symbol in self.positions:
            return False

        if len(self.positions) >= self.config.max_positions:
            return False

        if shares is None:
            shares = self.calculate_position_size(price)

        if shares <= 0:
            return False

        # Get volume from current data if available
        if avg_daily_volume is None and self.current_volume_data is not None:
            avg_daily_volume = self.current_volume_data.mean() if len(self.current_volume_data) > 0 else None

        # Calculate full cost breakdown
        cost_breakdown = self.get_total_transaction_cost(price, shares, avg_daily_volume)
        total_cost = price * shares + cost_breakdown.total_cost

        # Check if we have enough cash
        if total_cost > self.cash:
            # Reduce shares to fit
            available = self.cash - cost_breakdown.total_cost
            shares = int(available / price)
            if shares <= 0:
                return False
            # Recalculate with new share count
            cost_breakdown = self.get_total_transaction_cost(price, shares, avg_daily_volume)
            total_cost = price * shares + cost_breakdown.total_cost

        # Update cash
        self.cash -= total_cost

        # Create position
        position = Position(
            symbol=symbol,
            side=PositionSide.LONG,
            shares=shares,
            entry_price=price,
            entry_date=date,
            current_price=price
        )
        self.positions[symbol] = position

        # Create trade record with cost breakdown
        # Use commission + slippage for compatibility with base Trade class
        trade = Trade(
            entry_date=date,
            entry_price=price,
            shares=shares,
            side=PositionSide.LONG,
            commission=cost_breakdown.commission + cost_breakdown.spread_cost,
            slippage=cost_breakdown.slippage + cost_breakdown.market_impact
        )
        self.trades.append(trade)

        # Track cost breakdown
        if self.cost_config.track_cost_breakdown:
            self.cost_breakdowns.append(cost_breakdown)

        return True

    def exit_position(
        self,
        symbol: str,
        date: datetime,
        price: float,
        avg_daily_volume: Optional[float] = None
    ) -> bool:
        """
        Exit a position with advanced cost modeling

        Overrides base implementation to track cost breakdown.

        Args:
            symbol: Stock symbol
            date: Exit date
            price: Exit price
            avg_daily_volume: Average daily volume for impact calculation

        Returns:
            True if position was closed successfully
        """
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]

        # Get volume from current data if available
        if avg_daily_volume is None and self.current_volume_data is not None:
            avg_daily_volume = self.current_volume_data.mean() if len(self.current_volume_data) > 0 else None

        # Calculate cost breakdown
        cost_breakdown = self.get_total_transaction_cost(price, position.shares, avg_daily_volume)

        # Update cash
        if position.side == PositionSide.LONG:
            proceeds = price * position.shares - cost_breakdown.total_cost
            self.cash += proceeds
        else:  # SHORT
            cost = price * position.shares + cost_breakdown.total_cost
            margin_returned = position.entry_price * position.shares * self.config.margin_requirement
            self.cash += (margin_returned - cost)

        # Close the trade
        trade = self.trades[-1]
        # Combine exit costs with entry costs
        trade.close(
            date, price,
            commission=cost_breakdown.commission + cost_breakdown.spread_cost,
            slippage=cost_breakdown.slippage + cost_breakdown.market_impact
        )
        trade.mae = position.mae
        trade.mfe = position.mfe

        # Track cost breakdown
        if self.cost_config.track_cost_breakdown:
            self.cost_breakdowns.append(cost_breakdown)

        # Remove position
        del self.positions[symbol]

        return True

    def update(self, date: datetime, prices: Dict[str, float], volumes: Optional[Dict[str, float]] = None):
        """
        Update backtester with current prices and volumes

        Extended to track volume data for market impact calculations.

        Args:
            date: Current date
            prices: Dict of symbol -> price
            volumes: Dict of symbol -> volume (optional)
        """
        # Store volume data for cost calculations
        if volumes is not None:
            self.current_volume_data = pd.Series(volumes)

        # Call base update
        super().update(date, prices)

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func,
        benchmark_data: pd.DataFrame = None,
        symbol: str = 'symbol'
    ) -> CostAwareBacktestResults:
        """
        Run a backtest with advanced cost modeling

        Extends base run_backtest to:
            - Extract volume data for market impact
            - Calculate rolling volatility for slippage
            - Track detailed cost breakdowns

        Args:
            data: DataFrame with OHLCV data (must include 'Volume' column)
            strategy_func: Strategy function
            benchmark_data: Optional benchmark data
            symbol: Symbol name for position tracking

        Returns:
            CostAwareBacktestResults with cost analytics
        """
        self.reset()

        # Pre-calculate volatility for slippage estimation
        if 'Close' in data.columns:
            returns = data['Close'].pct_change()
            rolling_vol = returns.rolling(window=20).std()
        else:
            rolling_vol = pd.Series(0.02, index=data.index)

        # Extract volume data if available
        has_volume = 'Volume' in data.columns

        # Run strategy on each bar
        for idx, row in data.iterrows():
            date = idx if isinstance(idx, datetime) else pd.to_datetime(idx)

            # Update current volatility
            if idx in rolling_vol.index:
                vol = rolling_vol.loc[idx]
                self.current_volatility = vol if pd.notna(vol) and vol > 0 else 0.02

            # Update volume data (rolling window for ADV calculation)
            if has_volume and self.cost_config.use_volume_for_impact:
                vol_window = data.loc[:idx, 'Volume'].tail(20)
                self.current_volume_data = vol_window

            # Update positions
            volumes = {symbol: row['Volume']} if has_volume else None
            self.update(date, {symbol: row['Close']})

            # Run strategy
            strategy_func(self, date, row)

        # Close remaining positions
        final_date = data.index[-1]
        final_price = data.iloc[-1]['Close']
        for sym in list(self.positions.keys()):
            self.exit_position(sym, final_date, final_price)

        # Update final equity
        if len(self.positions) == 0:
            self.equity = self.cash
            self.equity_curve.append({
                'date': final_date,
                'equity': self.equity,
                'cash': self.cash,
                'positions': 0
            })

        # Calculate results with cost breakdown
        return self.calculate_cost_aware_results(data, benchmark_data)

    def calculate_cost_aware_results(
        self,
        data: pd.DataFrame,
        benchmark_data: pd.DataFrame = None
    ) -> CostAwareBacktestResults:
        """
        Calculate backtest results including detailed cost analytics

        Args:
            data: Historical price data
            benchmark_data: Optional benchmark data

        Returns:
            CostAwareBacktestResults with full cost breakdown
        """
        # Get base results
        base_results = super().calculate_results(data, benchmark_data)

        # Create extended results
        results = CostAwareBacktestResults(
            total_return=base_results.total_return,
            total_return_pct=base_results.total_return_pct,
            annualized_return=base_results.annualized_return,
            benchmark_return=base_results.benchmark_return,
            alpha=base_results.alpha,
            beta=base_results.beta,
            sharpe_ratio=base_results.sharpe_ratio,
            sortino_ratio=base_results.sortino_ratio,
            calmar_ratio=base_results.calmar_ratio,
            max_drawdown=base_results.max_drawdown,
            max_drawdown_duration=base_results.max_drawdown_duration,
            volatility=base_results.volatility,
            downside_deviation=base_results.downside_deviation,
            total_trades=base_results.total_trades,
            winning_trades=base_results.winning_trades,
            losing_trades=base_results.losing_trades,
            win_rate=base_results.win_rate,
            avg_win=base_results.avg_win,
            avg_loss=base_results.avg_loss,
            avg_win_pct=base_results.avg_win_pct,
            avg_loss_pct=base_results.avg_loss_pct,
            profit_factor=base_results.profit_factor,
            expectancy=base_results.expectancy,
            largest_win=base_results.largest_win,
            largest_loss=base_results.largest_loss,
            avg_trade_duration=base_results.avg_trade_duration,
            avg_mae=base_results.avg_mae,
            avg_mfe=base_results.avg_mfe,
            equity_curve=base_results.equity_curve,
            trades=base_results.trades
        )

        # Aggregate cost breakdowns
        if self.cost_breakdowns:
            results.total_commissions = sum(cb.commission for cb in self.cost_breakdowns)
            results.total_slippage = sum(cb.slippage for cb in self.cost_breakdowns)
            results.total_market_impact = sum(cb.market_impact for cb in self.cost_breakdowns)
            results.total_spread_costs = sum(cb.spread_cost for cb in self.cost_breakdowns)
            results.total_transaction_costs = sum(cb.total_cost for cb in self.cost_breakdowns)

            # Average cost in bps
            total_bps = sum(cb.total_bps for cb in self.cost_breakdowns)
            results.avg_cost_bps = total_bps / len(self.cost_breakdowns)

            # Store per-trade breakdown
            results.cost_breakdown_by_trade = [cb.to_dict() for cb in self.cost_breakdowns]

        return results


def create_cost_model(
    profile: Union[str, CostProfile] = 'retail',
    liquidity_tier: Union[str, LiquidityTier] = 'mid_cap'
) -> TransactionCostModel:
    """
    Factory function to create a cost model with common configurations

    Args:
        profile: Cost profile ('retail', 'institutional', 'hft')
        liquidity_tier: Default liquidity tier ('large_cap', 'mid_cap', 'small_cap', 'micro_cap')

    Returns:
        Configured TransactionCostModel

    Example:
        >>> model = create_cost_model('institutional', 'large_cap')
        >>> cost = model.total_transaction_cost(100.0, 10000, 5000000)
    """
    # Convert string to enum if needed
    if isinstance(profile, str):
        profile = CostProfile(profile.lower())
    if isinstance(liquidity_tier, str):
        liquidity_tier = LiquidityTier(liquidity_tier.lower())

    return TransactionCostModel(
        profile=profile,
        liquidity_tier=liquidity_tier
    )


def estimate_transaction_costs(
    price: float,
    shares: int,
    avg_daily_volume: float,
    profile: str = 'retail',
    volatility: float = 0.02
) -> Dict[str, float]:
    """
    Quick utility function to estimate transaction costs

    Args:
        price: Stock price
        shares: Number of shares
        avg_daily_volume: Average daily volume
        profile: Cost profile ('retail', 'institutional', 'hft')
        volatility: Daily volatility

    Returns:
        Dictionary with cost estimates

    Example:
        >>> costs = estimate_transaction_costs(150.0, 100, 10000000, 'retail')
        >>> print(f"Total cost: ${costs['total_cost']:.2f}")
    """
    model = create_cost_model(profile)
    breakdown = model.total_transaction_cost(
        price=price,
        shares=shares,
        avg_daily_volume=avg_daily_volume,
        volatility=volatility
    )
    return breakdown.to_dict()
