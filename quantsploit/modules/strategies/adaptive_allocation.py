"""
Adaptive Asset Allocation Strategy for Quantsploit

This module implements adaptive asset allocation strategies that dynamically
adjust portfolio weights based on market conditions, momentum, and risk signals.

Key Features:
- Momentum-based regime detection
- Volatility-scaled position sizing
- Correlation regime adaptation
- Dynamic rebalancing triggers
- Integration with backtesting framework

References:
    - Keller, W. & Butler, A. (2015). "Momentum and Markowitz: A Golden Combination"
    - Keller, W. & Keuning, J. (2016). "Protective Asset Allocation (PAA)"
    - Ilmanen, A. (2011). "Expected Returns"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"


@dataclass
class RegimeSignal:
    """
    Market regime signal.

    Attributes:
        date: Signal date
        regime: Current market regime
        confidence: Confidence in regime detection (0-1)
        momentum_score: Aggregate momentum score
        volatility_percentile: Current vol as percentile of history
        correlation_regime: High or low correlation environment
    """
    date: pd.Timestamp
    regime: MarketRegime
    confidence: float
    momentum_score: float
    volatility_percentile: float
    correlation_regime: str


@dataclass
class AdaptiveWeights:
    """
    Adaptive portfolio weights.

    Attributes:
        date: Weight calculation date
        weights: Dict mapping asset to weight
        regime: Current regime used for allocation
        safe_asset_allocation: Allocation to safe assets
        risk_asset_allocation: Allocation to risk assets
        rebalance_trigger: What triggered rebalance
    """
    date: pd.Timestamp
    weights: Dict[str, float]
    regime: MarketRegime
    safe_asset_allocation: float
    risk_asset_allocation: float
    rebalance_trigger: Optional[str] = None


class AdaptiveAssetAllocation:
    """
    Adaptive Asset Allocation Strategy.

    Dynamically adjusts portfolio weights based on momentum signals,
    volatility regimes, and correlation structures. Combines elements
    of tactical asset allocation with risk management.

    ★ Insight ─────────────────────────────────────
    Adaptive Allocation Philosophy:
    - Markets cycle between regimes; static allocation is suboptimal
    - Momentum signals help identify regime shifts early
    - Volatility scaling prevents outsized drawdowns
    - Correlation breakdown during crises requires defensive positioning
    - Safe assets provide crash protection when risk-off
    ─────────────────────────────────────────────────

    Example:
        >>> strategy = AdaptiveAssetAllocation(returns_df, safe_assets=['TLT', 'GLD'])
        >>> weights = strategy.calculate_adaptive_weights()
        >>> results = strategy.run_backtest()

    Attributes:
        returns: DataFrame of asset returns
        safe_assets: List of safe haven assets
        risk_assets: List of risk assets
        lookback: Momentum lookback period
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        safe_assets: Optional[List[str]] = None,
        risk_assets: Optional[List[str]] = None,
        momentum_lookback: int = 12,  # Months
        volatility_lookback: int = 20,  # Days
        rebalance_frequency: int = 21,  # Days (monthly)
        max_safe_allocation: float = 1.0,
        min_safe_allocation: float = 0.0
    ):
        """
        Initialize Adaptive Asset Allocation Strategy.

        Args:
            returns: DataFrame of asset returns (daily)
            safe_assets: List of safe haven assets (e.g., bonds, gold)
            risk_assets: List of risk assets (e.g., equities)
            momentum_lookback: Months for momentum calculation
            volatility_lookback: Days for volatility calculation
            rebalance_frequency: Days between rebalances
            max_safe_allocation: Maximum allocation to safe assets
            min_safe_allocation: Minimum allocation to safe assets
        """
        self.returns = returns.copy()
        self.assets = list(returns.columns)

        # Classify assets
        if safe_assets is None:
            self.safe_assets = []
        else:
            self.safe_assets = [a for a in safe_assets if a in self.assets]

        if risk_assets is None:
            self.risk_assets = [a for a in self.assets if a not in self.safe_assets]
        else:
            self.risk_assets = [a for a in risk_assets if a in self.assets]

        self.momentum_lookback = momentum_lookback
        self.volatility_lookback = volatility_lookback
        self.rebalance_frequency = rebalance_frequency
        self.max_safe_allocation = max_safe_allocation
        self.min_safe_allocation = min_safe_allocation

        # Calculate cumulative prices for momentum
        self.prices = (1 + self.returns).cumprod()

    def calculate_momentum_score(
        self,
        lookback_months: Optional[int] = None,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, float]:
        """
        Calculate momentum scores for all assets.

        Uses 12-1 month momentum (skip most recent month).

        Args:
            lookback_months: Momentum lookback period
            as_of_date: Calculate as of this date

        Returns:
            Dict mapping asset to momentum score
        """
        if lookback_months is None:
            lookback_months = self.momentum_lookback

        lookback_days = lookback_months * 21  # Approximate trading days

        if as_of_date is not None:
            prices = self.prices.loc[:as_of_date]
        else:
            prices = self.prices

        if len(prices) < lookback_days + 21:
            return {asset: 0.0 for asset in self.assets}

        # 12-1 month momentum (skip most recent month)
        current = prices.iloc[-21]  # Price 1 month ago
        past = prices.iloc[-(lookback_days + 21)]  # Price lookback+1 months ago

        momentum = {}
        for asset in self.assets:
            if past[asset] > 0:
                momentum[asset] = (current[asset] / past[asset] - 1) * 100
            else:
                momentum[asset] = 0.0

        return momentum

    def calculate_volatility_regime(
        self,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> Tuple[float, float]:
        """
        Calculate current volatility and its percentile.

        Args:
            as_of_date: Calculate as of this date

        Returns:
            Tuple of (current_vol, vol_percentile)
        """
        if as_of_date is not None:
            returns = self.returns.loc[:as_of_date]
        else:
            returns = self.returns

        # Portfolio volatility (equal weight for simplicity)
        portfolio_returns = returns[self.risk_assets].mean(axis=1) if self.risk_assets else returns.mean(axis=1)

        # Current volatility (annualized)
        current_vol = portfolio_returns.tail(self.volatility_lookback).std() * np.sqrt(252)

        # Historical volatility distribution
        rolling_vol = portfolio_returns.rolling(self.volatility_lookback).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) == 0:
            return current_vol, 50.0

        # Percentile
        percentile = (rolling_vol < current_vol).mean() * 100

        return current_vol, percentile

    def calculate_correlation_regime(
        self,
        as_of_date: Optional[pd.Timestamp] = None,
        lookback: int = 60
    ) -> str:
        """
        Detect correlation regime (high or low correlation environment).

        Args:
            as_of_date: Calculate as of this date
            lookback: Days for correlation calculation

        Returns:
            'high' or 'low' correlation regime
        """
        if as_of_date is not None:
            returns = self.returns.loc[:as_of_date]
        else:
            returns = self.returns

        if len(returns) < lookback:
            return 'normal'

        recent_returns = returns.tail(lookback)
        corr_matrix = recent_returns.corr()

        # Average pairwise correlation
        n = len(self.assets)
        if n < 2:
            return 'normal'

        avg_corr = (corr_matrix.sum().sum() - n) / (n * (n - 1))

        # High correlation threshold (crisis indicator)
        if avg_corr > 0.6:
            return 'high'
        elif avg_corr < 0.3:
            return 'low'
        else:
            return 'normal'

    def detect_regime(
        self,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> RegimeSignal:
        """
        Detect current market regime.

        Args:
            as_of_date: Detect regime as of this date

        Returns:
            RegimeSignal with regime classification
        """
        if as_of_date is None:
            as_of_date = self.returns.index[-1]

        # Get momentum scores
        momentum = self.calculate_momentum_score(as_of_date=as_of_date)

        # Average risk asset momentum
        risk_momentum = np.mean([momentum.get(a, 0) for a in self.risk_assets]) if self.risk_assets else 0

        # Get volatility
        current_vol, vol_percentile = self.calculate_volatility_regime(as_of_date)

        # Get correlation regime
        corr_regime = self.calculate_correlation_regime(as_of_date)

        # Determine regime
        if risk_momentum > 5 and vol_percentile < 70:
            regime = MarketRegime.BULL
            confidence = min(risk_momentum / 20, 1.0)
        elif risk_momentum < -5 and vol_percentile > 50:
            regime = MarketRegime.BEAR
            confidence = min(abs(risk_momentum) / 20, 1.0)
        elif vol_percentile > 80:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = (vol_percentile - 50) / 50
        elif vol_percentile < 20:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = (50 - vol_percentile) / 50
        elif risk_momentum > 0:
            regime = MarketRegime.RISK_ON
            confidence = 0.5 + risk_momentum / 40
        else:
            regime = MarketRegime.RISK_OFF
            confidence = 0.5 + abs(risk_momentum) / 40

        return RegimeSignal(
            date=as_of_date,
            regime=regime,
            confidence=min(confidence, 1.0),
            momentum_score=risk_momentum,
            volatility_percentile=vol_percentile,
            correlation_regime=corr_regime
        )

    def calculate_base_weights(
        self,
        method: str = 'momentum',
        as_of_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, float]:
        """
        Calculate base portfolio weights before regime adjustment.

        Args:
            method: Weight calculation method ('momentum', 'inverse_vol', 'equal')
            as_of_date: Calculate as of this date

        Returns:
            Dict mapping asset to weight
        """
        if method == 'equal':
            return {asset: 1.0 / len(self.assets) for asset in self.assets}

        elif method == 'inverse_vol':
            if as_of_date is not None:
                returns = self.returns.loc[:as_of_date]
            else:
                returns = self.returns

            vols = returns.tail(self.volatility_lookback).std()
            inv_vols = 1 / vols.replace(0, np.nan).fillna(vols.max())
            weights = inv_vols / inv_vols.sum()
            return weights.to_dict()

        elif method == 'momentum':
            momentum = self.calculate_momentum_score(as_of_date=as_of_date)

            # Only include positive momentum assets
            positive_mom = {k: max(v, 0) for k, v in momentum.items()}
            total = sum(positive_mom.values())

            if total == 0:
                # Fall back to equal weight among safe assets
                if self.safe_assets:
                    return {a: 1.0 / len(self.safe_assets) if a in self.safe_assets else 0
                            for a in self.assets}
                else:
                    return {asset: 1.0 / len(self.assets) for asset in self.assets}

            return {k: v / total for k, v in positive_mom.items()}

        else:
            raise ValueError(f"Unknown method: {method}")

    def calculate_adaptive_weights(
        self,
        as_of_date: Optional[pd.Timestamp] = None,
        base_method: str = 'momentum'
    ) -> AdaptiveWeights:
        """
        Calculate regime-adaptive portfolio weights.

        Args:
            as_of_date: Calculate as of this date
            base_method: Base weight calculation method

        Returns:
            AdaptiveWeights with final portfolio allocation
        """
        if as_of_date is None:
            as_of_date = self.returns.index[-1]

        # Detect regime
        regime = self.detect_regime(as_of_date)

        # Calculate base weights
        base_weights = self.calculate_base_weights(base_method, as_of_date)

        # Adjust based on regime
        safe_allocation = self._regime_safe_allocation(regime)

        # Split allocation
        risk_allocation = 1 - safe_allocation

        # Distribute within each category
        final_weights = {}

        # Safe assets
        if self.safe_assets and safe_allocation > 0:
            safe_sum = sum(base_weights.get(a, 0) for a in self.safe_assets)
            if safe_sum == 0:
                safe_sum = len(self.safe_assets)
                for a in self.safe_assets:
                    final_weights[a] = safe_allocation / len(self.safe_assets)
            else:
                for a in self.safe_assets:
                    final_weights[a] = safe_allocation * (base_weights.get(a, 0) / safe_sum)

        # Risk assets
        if self.risk_assets and risk_allocation > 0:
            risk_sum = sum(base_weights.get(a, 0) for a in self.risk_assets)
            if risk_sum == 0:
                risk_sum = len(self.risk_assets)
                for a in self.risk_assets:
                    final_weights[a] = risk_allocation / len(self.risk_assets)
            else:
                for a in self.risk_assets:
                    final_weights[a] = risk_allocation * (base_weights.get(a, 0) / risk_sum)

        # Ensure all assets have weight (even if 0)
        for asset in self.assets:
            if asset not in final_weights:
                final_weights[asset] = 0.0

        return AdaptiveWeights(
            date=as_of_date,
            weights=final_weights,
            regime=regime.regime,
            safe_asset_allocation=safe_allocation,
            risk_asset_allocation=risk_allocation,
            rebalance_trigger='scheduled'
        )

    def _regime_safe_allocation(self, regime: RegimeSignal) -> float:
        """Determine safe asset allocation based on regime."""
        base_safe = 0.3  # Default safe allocation

        if regime.regime == MarketRegime.BULL:
            safe = self.min_safe_allocation + 0.1
        elif regime.regime == MarketRegime.BEAR:
            safe = self.max_safe_allocation - 0.1
        elif regime.regime == MarketRegime.HIGH_VOLATILITY:
            safe = 0.6 + 0.2 * regime.confidence
        elif regime.regime == MarketRegime.LOW_VOLATILITY:
            safe = 0.2
        elif regime.regime == MarketRegime.RISK_ON:
            safe = 0.25
        elif regime.regime == MarketRegime.RISK_OFF:
            safe = 0.5 + 0.2 * regime.confidence
        else:
            safe = base_safe

        # Apply correlation adjustment
        if regime.correlation_regime == 'high':
            safe = min(safe + 0.1, self.max_safe_allocation)

        return np.clip(safe, self.min_safe_allocation, self.max_safe_allocation)

    def check_rebalance_trigger(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> Tuple[bool, str]:
        """
        Check if rebalancing is triggered.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            threshold: Drift threshold for rebalancing

        Returns:
            Tuple of (should_rebalance, reason)
        """
        max_drift = 0
        max_drift_asset = None

        for asset in self.assets:
            current = current_weights.get(asset, 0)
            target = target_weights.get(asset, 0)
            drift = abs(current - target)

            if drift > max_drift:
                max_drift = drift
                max_drift_asset = asset

        if max_drift > threshold:
            return True, f"drift_{max_drift_asset}_{max_drift:.2%}"

        return False, None

    def run_backtest(
        self,
        initial_capital: float = 100000.0,
        base_method: str = 'momentum',
        rebalance_frequency: Optional[int] = None,
        transaction_cost: float = 0.001
    ) -> Dict:
        """
        Backtest the adaptive allocation strategy.

        Args:
            initial_capital: Starting capital
            base_method: Base weight method
            rebalance_frequency: Days between rebalances
            transaction_cost: Transaction cost as percentage

        Returns:
            Dictionary with backtest results
        """
        if rebalance_frequency is None:
            rebalance_frequency = self.rebalance_frequency

        # Minimum lookback needed
        min_lookback = max(self.momentum_lookback * 21 + 21, self.volatility_lookback + 60)

        if len(self.returns) < min_lookback:
            return {
                'error': 'Insufficient data',
                'required_days': min_lookback,
                'available_days': len(self.returns)
            }

        # Start after lookback
        start_idx = min_lookback
        portfolio_value = initial_capital
        cash = initial_capital
        positions = {asset: 0.0 for asset in self.assets}

        equity_curve = []
        weight_history = []
        regime_history = []
        trades = []

        last_rebalance = 0

        for i in range(start_idx, len(self.returns)):
            date = self.returns.index[i]
            daily_returns = self.returns.iloc[i]

            # Update positions with daily returns
            for asset in self.assets:
                if positions[asset] > 0:
                    positions[asset] *= (1 + daily_returns[asset])

            # Calculate current portfolio value
            portfolio_value = sum(positions.values()) + cash

            # Check for rebalance
            days_since_rebalance = i - last_rebalance

            if days_since_rebalance >= rebalance_frequency:
                # Calculate target weights
                target = self.calculate_adaptive_weights(as_of_date=date, base_method=base_method)

                # Execute rebalance
                target_positions = {
                    asset: portfolio_value * weight
                    for asset, weight in target.weights.items()
                }

                # Calculate turnover and costs
                turnover = 0
                for asset in self.assets:
                    turnover += abs(target_positions[asset] - positions[asset])

                cost = turnover * transaction_cost
                portfolio_value -= cost

                # Update positions
                positions = target_positions.copy()
                cash = 0

                # Record
                weight_history.append({
                    'date': date,
                    **target.weights,
                    'safe_allocation': target.safe_asset_allocation,
                    'risk_allocation': target.risk_asset_allocation
                })

                regime_history.append({
                    'date': date,
                    'regime': target.regime.value,
                })

                trades.append({
                    'date': date,
                    'turnover': turnover,
                    'cost': cost
                })

                last_rebalance = i

            equity_curve.append({
                'date': date,
                'value': portfolio_value
            })

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        equity_df['returns'] = equity_df['value'].pct_change()

        total_return = (portfolio_value - initial_capital) / initial_capital
        returns = equity_df['returns'].dropna()

        # Annualized return
        n_years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Sharpe ratio
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Max drawdown
        equity = equity_df['value']
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_drawdown = drawdown.min()

        # Regime stats
        regime_df = pd.DataFrame(regime_history)
        regime_counts = regime_df['regime'].value_counts().to_dict() if len(regime_df) > 0 else {}

        return {
            'initial_capital': initial_capital,
            'final_value': portfolio_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'n_rebalances': len(trades),
            'total_turnover': sum(t['turnover'] for t in trades),
            'total_costs': sum(t['cost'] for t in trades),
            'regime_counts': regime_counts,
            'equity_curve': equity_df,
            'weight_history': pd.DataFrame(weight_history) if weight_history else pd.DataFrame(),
            'trades': trades
        }

    def generate_signals_report(self) -> pd.DataFrame:
        """
        Generate a report of all regime signals over the backtest period.

        Returns:
            DataFrame with regime signals and recommendations
        """
        min_lookback = max(self.momentum_lookback * 21 + 21, self.volatility_lookback + 60)

        signals = []
        for i in range(min_lookback, len(self.returns), self.rebalance_frequency):
            date = self.returns.index[i]
            regime = self.detect_regime(as_of_date=date)
            weights = self.calculate_adaptive_weights(as_of_date=date)

            signals.append({
                'date': date,
                'regime': regime.regime.value,
                'confidence': regime.confidence,
                'momentum_score': regime.momentum_score,
                'vol_percentile': regime.volatility_percentile,
                'correlation_regime': regime.correlation_regime,
                'safe_allocation': weights.safe_asset_allocation,
                'risk_allocation': weights.risk_asset_allocation
            })

        return pd.DataFrame(signals)


class ProtectiveAssetAllocation:
    """
    Protective Asset Allocation (PAA) Implementation.

    Based on Keller & Keuning (2016), this strategy uses breadth momentum
    to determine the allocation between risk and safe assets.

    ★ Insight ─────────────────────────────────────
    PAA Key Concepts:
    - Breadth momentum: % of assets with positive momentum
    - Higher breadth = lower protection (more risk assets)
    - Lower breadth = higher protection (more safe assets)
    - Protection factor controls sensitivity
    ─────────────────────────────────────────────────
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_assets: List[str],
        safe_asset: str,
        protection_factor: int = 2,  # 0, 1, or 2 (low, medium, high protection)
        momentum_lookback: int = 12,  # Months
        top_n: Optional[int] = None  # Select top N momentum assets
    ):
        """
        Initialize Protective Asset Allocation.

        Args:
            returns: DataFrame of asset returns
            risk_assets: List of risk asset names
            safe_asset: Safe asset name (e.g., 'BIL', 'SHY')
            protection_factor: 0=low, 1=medium, 2=high protection
            momentum_lookback: Months for momentum calculation
            top_n: If set, only invest in top N momentum risk assets
        """
        self.returns = returns.copy()
        self.risk_assets = [a for a in risk_assets if a in returns.columns]
        self.safe_asset = safe_asset if safe_asset in returns.columns else None
        self.protection_factor = protection_factor
        self.momentum_lookback = momentum_lookback
        self.top_n = top_n or len(self.risk_assets)

        # Calculate prices
        self.prices = (1 + self.returns).cumprod()

    def calculate_breadth(
        self,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Calculate breadth momentum (% of assets with positive momentum).

        Args:
            as_of_date: Calculate as of this date

        Returns:
            Breadth as percentage (0-1)
        """
        lookback_days = self.momentum_lookback * 21

        if as_of_date is not None:
            prices = self.prices.loc[:as_of_date]
        else:
            prices = self.prices

        if len(prices) < lookback_days:
            return 0.5  # Default

        current = prices.iloc[-1]
        past = prices.iloc[-lookback_days]

        n_positive = 0
        for asset in self.risk_assets:
            if past[asset] > 0:
                mom = current[asset] / past[asset] - 1
                if mom > 0:
                    n_positive += 1

        return n_positive / len(self.risk_assets) if self.risk_assets else 0.5

    def calculate_bond_fraction(self, breadth: float) -> float:
        """
        Calculate safe asset fraction based on breadth.

        Uses PAA formula: BF = (N - n) / N * protection_multiplier

        Args:
            breadth: Breadth momentum (0-1)

        Returns:
            Bond/safe asset fraction (0-1)
        """
        n = len(self.risk_assets)

        # Protection multipliers
        multipliers = {0: 0, 1: 1, 2: 2}
        mult = multipliers.get(self.protection_factor, 1)

        # Number of assets with negative momentum
        n_negative = int((1 - breadth) * n)

        # Bond fraction
        bf = min(1, n_negative / n * mult)

        return bf

    def calculate_weights(
        self,
        as_of_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, float]:
        """
        Calculate PAA weights.

        Args:
            as_of_date: Calculate as of this date

        Returns:
            Dict mapping asset to weight
        """
        lookback_days = self.momentum_lookback * 21

        if as_of_date is not None:
            prices = self.prices.loc[:as_of_date]
        else:
            prices = self.prices

        if len(prices) < lookback_days:
            # Equal weight among risk assets
            weight = 1.0 / (len(self.risk_assets) + (1 if self.safe_asset else 0))
            weights = {a: weight for a in self.risk_assets}
            if self.safe_asset:
                weights[self.safe_asset] = weight
            return weights

        # Calculate breadth and bond fraction
        breadth = self.calculate_breadth(as_of_date)
        bond_fraction = self.calculate_bond_fraction(breadth)

        # Calculate momentum for ranking
        current = prices.iloc[-1]
        past = prices.iloc[-lookback_days]

        momentum = {}
        for asset in self.risk_assets:
            if past[asset] > 0:
                momentum[asset] = current[asset] / past[asset] - 1
            else:
                momentum[asset] = -999

        # Select top N by momentum (only positive momentum)
        sorted_assets = sorted(momentum.items(), key=lambda x: x[1], reverse=True)
        selected = [a for a, m in sorted_assets[:self.top_n] if m > 0]

        # Calculate weights
        weights = {a: 0.0 for a in self.risk_assets}
        if self.safe_asset:
            weights[self.safe_asset] = 0.0

        # Risk allocation among selected assets
        risk_allocation = 1 - bond_fraction
        if selected:
            risk_per_asset = risk_allocation / len(selected)
            for asset in selected:
                weights[asset] = risk_per_asset

        # Safe asset allocation
        if self.safe_asset:
            weights[self.safe_asset] = bond_fraction

        return weights

    def run_backtest(
        self,
        initial_capital: float = 100000.0,
        rebalance_frequency: int = 21,  # Monthly
        transaction_cost: float = 0.001
    ) -> Dict:
        """
        Backtest PAA strategy.

        Args:
            initial_capital: Starting capital
            rebalance_frequency: Days between rebalances
            transaction_cost: Transaction cost percentage

        Returns:
            Dictionary with backtest results
        """
        min_lookback = self.momentum_lookback * 21 + 5

        if len(self.returns) < min_lookback:
            return {'error': 'Insufficient data'}

        all_assets = self.risk_assets + ([self.safe_asset] if self.safe_asset else [])
        positions = {a: 0.0 for a in all_assets}
        portfolio_value = initial_capital

        equity_curve = []
        weight_history = []

        last_rebalance = 0

        for i in range(min_lookback, len(self.returns)):
            date = self.returns.index[i]
            daily_returns = self.returns.iloc[i]

            # Update positions
            for asset in all_assets:
                if asset in daily_returns.index:
                    positions[asset] *= (1 + daily_returns[asset])

            portfolio_value = sum(positions.values())

            # Rebalance
            if i - last_rebalance >= rebalance_frequency:
                target_weights = self.calculate_weights(as_of_date=date)

                # Calculate turnover
                turnover = 0
                for asset in all_assets:
                    current = positions[asset] / portfolio_value if portfolio_value > 0 else 0
                    target = target_weights.get(asset, 0)
                    turnover += abs(target - current) * portfolio_value

                cost = turnover * transaction_cost
                portfolio_value -= cost

                # Update positions
                for asset in all_assets:
                    positions[asset] = portfolio_value * target_weights.get(asset, 0)

                weight_history.append({
                    'date': date,
                    'bond_fraction': target_weights.get(self.safe_asset, 0) if self.safe_asset else 0,
                    **target_weights
                })

                last_rebalance = i

            equity_curve.append({
                'date': date,
                'value': portfolio_value
            })

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve).set_index('date')
        equity_df['returns'] = equity_df['value'].pct_change()

        total_return = (portfolio_value - initial_capital) / initial_capital
        returns = equity_df['returns'].dropna()

        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        peak = equity_df['value'].expanding().max()
        drawdown = (equity_df['value'] - peak) / peak
        max_drawdown = drawdown.min()

        return {
            'initial_capital': initial_capital,
            'final_value': portfolio_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_df,
            'weight_history': pd.DataFrame(weight_history) if weight_history else pd.DataFrame()
        }
