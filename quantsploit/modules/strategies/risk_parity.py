"""
Risk Parity Strategy for Quantsploit

This module implements Risk Parity portfolio allocation strategies that aim
to equalize the risk contribution of each asset in the portfolio, rather
than equalizing dollar allocations.

Key Features:
- Inverse volatility weighting (simple risk parity)
- Equal Risk Contribution (ERC) optimization
- Hierarchical Risk Parity (HRP) using clustering
- Volatility targeting and leverage management
- Dynamic rebalancing with transaction cost awareness
- Integration with walk-forward optimization

References:
    - Qian, E. (2005). "Risk Parity Portfolios"
    - Maillard et al. (2010). "On the Properties of ERC Portfolios"
    - Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import warnings

logger = logging.getLogger(__name__)


class RiskParityMethod(Enum):
    """Risk parity allocation methods"""
    INVERSE_VOLATILITY = "inverse_volatility"
    EQUAL_RISK_CONTRIBUTION = "erc"
    HIERARCHICAL_RISK_PARITY = "hrp"


@dataclass
class RiskParityResult:
    """
    Results from risk parity allocation.

    Attributes:
        weights: Optimal portfolio weights
        risk_contributions: Risk contribution of each asset
        risk_contribution_pct: Risk contribution as percentage
        portfolio_volatility: Total portfolio volatility
        effective_n_assets: Effective number of assets (diversification)
        concentration_ratio: Herfindahl-Hirschman index of risk contributions
        method: Method used for allocation
        converged: Whether optimization converged (for ERC)
    """
    weights: np.ndarray
    risk_contributions: np.ndarray
    risk_contribution_pct: np.ndarray
    portfolio_volatility: float
    effective_n_assets: float
    concentration_ratio: float
    method: str
    converged: bool = True
    asset_names: Optional[List[str]] = None

    def summary(self) -> str:
        """Return a formatted summary string."""
        names = self.asset_names if self.asset_names else [f"Asset_{i}" for i in range(len(self.weights))]
        lines = [
            f"Risk Parity Allocation ({self.method})",
            "=" * 50,
            f"Portfolio Volatility: {self.portfolio_volatility:.2%}",
            f"Effective N Assets: {self.effective_n_assets:.2f}",
            f"Concentration Ratio: {self.concentration_ratio:.4f}",
            "",
            "Asset Allocations:",
            "-" * 50,
            f"{'Asset':<15} {'Weight':>10} {'Risk Contrib':>12} {'% of Risk':>10}"
        ]

        for i, name in enumerate(names):
            lines.append(
                f"{name:<15} {self.weights[i]:>10.2%} "
                f"{self.risk_contributions[i]:>12.4f} {self.risk_contribution_pct[i]:>10.1%}"
            )

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        names = self.asset_names if self.asset_names else [f"Asset_{i}" for i in range(len(self.weights))]
        return pd.DataFrame({
            'Asset': names,
            'Weight': self.weights,
            'Risk_Contribution': self.risk_contributions,
            'Risk_Contribution_Pct': self.risk_contribution_pct
        }).set_index('Asset')


class RiskParityStrategy:
    """
    Risk Parity Portfolio Strategy.

    Risk parity allocates capital such that each asset contributes equally
    to the total portfolio risk. This typically results in overweighting
    low-volatility assets and underweighting high-volatility assets.

    ★ Insight ─────────────────────────────────────
    Why Risk Parity?
    - Traditional 60/40 portfolios have ~90% equity risk
    - Risk parity balances risk, not just dollars
    - Works well in diversified macro environments
    - Can be leveraged to target specific volatility
    ─────────────────────────────────────────────────

    Example:
        >>> strategy = RiskParityStrategy(returns_df)
        >>> result = strategy.calculate_weights(method='erc')
        >>> print(result.summary())

    Attributes:
        returns: Asset returns DataFrame
        n_assets: Number of assets
        asset_names: Names of assets
        covariance: Covariance matrix
        volatilities: Individual asset volatilities
        correlation: Correlation matrix
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02,
        annualization_factor: int = 252
    ):
        """
        Initialize Risk Parity Strategy.

        Args:
            returns: DataFrame of asset returns (columns = assets)
            risk_free_rate: Annual risk-free rate
            annualization_factor: Factor to annualize returns/vol
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

        self.n_assets = returns.shape[1]
        self.asset_names = list(returns.columns)

        # Calculate statistics
        self.covariance = returns.cov() * annualization_factor
        self.volatilities = returns.std() * np.sqrt(annualization_factor)
        self.correlation = returns.corr()
        self.expected_returns = returns.mean() * annualization_factor

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility."""
        return np.sqrt(weights @ self.covariance.values @ weights)

    def _marginal_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate marginal risk contribution of each asset.

        MRC_i = (Cov @ w)_i / sigma_p
        """
        portfolio_vol = self._portfolio_volatility(weights)
        if portfolio_vol == 0:
            return np.zeros(self.n_assets)

        marginal_contrib = self.covariance.values @ weights / portfolio_vol
        return marginal_contrib

    def _risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """
        Calculate total risk contribution of each asset.

        RC_i = w_i * MRC_i
        """
        mrc = self._marginal_risk_contributions(weights)
        return weights * mrc

    def inverse_volatility_weights(self) -> RiskParityResult:
        """
        Calculate inverse volatility weights (simple risk parity).

        Weights are proportional to 1/volatility. This is a fast
        approximation that works well when correlations are similar.

        Returns:
            RiskParityResult with inverse volatility weights
        """
        inv_vol = 1 / self.volatilities.values
        weights = inv_vol / inv_vol.sum()

        risk_contrib = self._risk_contributions(weights)
        portfolio_vol = self._portfolio_volatility(weights)

        return self._create_result(weights, risk_contrib, portfolio_vol, 'inverse_volatility')

    def equal_risk_contribution_weights(
        self,
        target_risk: Optional[np.ndarray] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0
    ) -> RiskParityResult:
        """
        Calculate Equal Risk Contribution (ERC) weights.

        Optimizes weights such that each asset contributes equally to
        total portfolio risk: RC_i = RC_j for all i, j.

        The optimization minimizes:
            sum_i sum_j (RC_i - RC_j)^2

        Args:
            target_risk: Optional target risk contributions (default: equal)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset

        Returns:
            RiskParityResult with ERC optimal weights
        """
        if target_risk is None:
            target_risk = np.ones(self.n_assets) / self.n_assets

        def objective(weights):
            """Minimize squared differences in risk contributions."""
            rc = self._risk_contributions(weights)
            total_risk = rc.sum()

            if total_risk <= 0:
                return 1e10

            rc_pct = rc / total_risk
            # Sum of squared deviations from target
            return np.sum((rc_pct - target_risk) ** 2)

        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # Bounds
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]

        # Initial guess: inverse volatility
        x0 = self.inverse_volatility_weights().weights

        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )

        weights = result.x
        risk_contrib = self._risk_contributions(weights)
        portfolio_vol = self._portfolio_volatility(weights)

        return self._create_result(
            weights, risk_contrib, portfolio_vol, 'erc',
            converged=result.success
        )

    def hierarchical_risk_parity_weights(
        self,
        linkage_method: str = 'single'
    ) -> RiskParityResult:
        """
        Calculate Hierarchical Risk Parity (HRP) weights.

        HRP uses hierarchical clustering to group similar assets and
        allocates recursively through the hierarchy. This produces
        more stable allocations and handles estimation error better.

        Algorithm:
        1. Tree clustering using correlation distance
        2. Quasi-diagonalization (reorder to put similar assets together)
        3. Recursive bisection to allocate weights

        ★ Insight ─────────────────────────────────────
        HRP addresses two weaknesses of traditional optimization:
        - No matrix inversion needed (numerically stable)
        - Uses correlation structure via clustering (robust)
        - Out-of-sample performance often beats Markowitz
        ─────────────────────────────────────────────────

        Args:
            linkage_method: Clustering method ('single', 'complete', 'average', 'ward')

        Returns:
            RiskParityResult with HRP weights
        """
        # Step 1: Compute correlation distance matrix
        corr = self.correlation.values
        dist = np.sqrt(0.5 * (1 - corr))

        # Step 2: Hierarchical clustering
        condensed_dist = squareform(dist, checks=False)
        link = linkage(condensed_dist, method=linkage_method)

        # Step 3: Quasi-diagonalization (get sorted indices)
        sorted_idx = leaves_list(link)

        # Step 4: Recursive bisection
        weights = self._recursive_bisection(sorted_idx)

        risk_contrib = self._risk_contributions(weights)
        portfolio_vol = self._portfolio_volatility(weights)

        return self._create_result(weights, risk_contrib, portfolio_vol, 'hrp')

    def _recursive_bisection(self, sorted_idx: np.ndarray) -> np.ndarray:
        """
        Perform recursive bisection for HRP.

        Recursively splits the sorted indices and allocates based on
        cluster variance.
        """
        weights = np.ones(self.n_assets)
        clusters = [sorted_idx]

        while len(clusters) > 0:
            # Split each cluster
            new_clusters = []

            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                # Split in half
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Calculate cluster variances
                var_left = self._cluster_variance(left)
                var_right = self._cluster_variance(right)

                # Allocate inversely proportional to variance
                alpha = 1 - var_left / (var_left + var_right)

                # Update weights
                weights[left] *= alpha
                weights[right] *= (1 - alpha)

                # Add sub-clusters if they can be split further
                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        return weights / weights.sum()

    def _cluster_variance(self, indices: np.ndarray) -> float:
        """Calculate variance of a cluster using inverse variance portfolio."""
        if len(indices) == 1:
            return self.covariance.values[indices[0], indices[0]]

        # Sub-covariance matrix
        sub_cov = self.covariance.values[np.ix_(indices, indices)]

        # Inverse variance weights within cluster
        diag = np.diag(sub_cov)
        if np.any(diag <= 0):
            return np.mean(diag)

        inv_diag = 1 / diag
        weights = inv_diag / inv_diag.sum()

        # Cluster variance
        return weights @ sub_cov @ weights

    def _create_result(
        self,
        weights: np.ndarray,
        risk_contrib: np.ndarray,
        portfolio_vol: float,
        method: str,
        converged: bool = True
    ) -> RiskParityResult:
        """Create RiskParityResult from components."""
        total_risk = risk_contrib.sum()
        risk_contrib_pct = risk_contrib / total_risk if total_risk > 0 else risk_contrib

        # Effective N (diversification measure)
        effective_n = 1 / np.sum(risk_contrib_pct ** 2)

        # Concentration ratio (HHI)
        concentration = np.sum(risk_contrib_pct ** 2)

        return RiskParityResult(
            weights=weights,
            risk_contributions=risk_contrib,
            risk_contribution_pct=risk_contrib_pct,
            portfolio_volatility=portfolio_vol,
            effective_n_assets=effective_n,
            concentration_ratio=concentration,
            method=method,
            converged=converged,
            asset_names=self.asset_names
        )

    def calculate_weights(
        self,
        method: Union[str, RiskParityMethod] = 'erc',
        **kwargs
    ) -> RiskParityResult:
        """
        Calculate risk parity weights using specified method.

        Args:
            method: 'inverse_volatility', 'erc', or 'hrp'
            **kwargs: Additional arguments for specific method

        Returns:
            RiskParityResult with optimal weights
        """
        if isinstance(method, RiskParityMethod):
            method = method.value

        method = method.lower()

        if method == 'inverse_volatility':
            return self.inverse_volatility_weights()
        elif method == 'erc':
            return self.equal_risk_contribution_weights(**kwargs)
        elif method == 'hrp':
            return self.hierarchical_risk_parity_weights(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def compare_methods(self) -> pd.DataFrame:
        """
        Compare all risk parity methods.

        Returns:
            DataFrame comparing weights and metrics across methods
        """
        results = {
            'Inverse Vol': self.inverse_volatility_weights(),
            'ERC': self.equal_risk_contribution_weights(),
            'HRP': self.hierarchical_risk_parity_weights()
        }

        comparison = []
        for name, result in results.items():
            for i, asset in enumerate(self.asset_names):
                comparison.append({
                    'Method': name,
                    'Asset': asset,
                    'Weight': result.weights[i],
                    'Risk_Contribution_Pct': result.risk_contribution_pct[i]
                })

        return pd.DataFrame(comparison)

    def backtest(
        self,
        rebalance_frequency: int = 21,
        lookback_window: int = 252,
        method: str = 'erc',
        transaction_cost: float = 0.001,
        target_volatility: Optional[float] = None,
        max_leverage: float = 2.0
    ) -> Dict:
        """
        Backtest risk parity strategy with periodic rebalancing.

        Args:
            rebalance_frequency: Days between rebalances
            lookback_window: Days for covariance estimation
            method: Risk parity method
            transaction_cost: Transaction cost as percentage
            target_volatility: Optional volatility target (enables leverage)
            max_leverage: Maximum leverage if targeting volatility

        Returns:
            Dictionary with backtest results
        """
        dates = self.returns.index
        n_days = len(dates)

        # Initialize tracking
        portfolio_returns = []
        weights_history = []
        rebalance_dates = []
        turnover_history = []

        current_weights = np.zeros(self.n_assets)

        for i in range(lookback_window, n_days):
            date = dates[i]
            day_returns = self.returns.iloc[i].values

            # Check if rebalance day
            if (i - lookback_window) % rebalance_frequency == 0:
                # Get lookback returns
                lookback_returns = self.returns.iloc[i - lookback_window:i]

                # Calculate new weights
                temp_strategy = RiskParityStrategy(
                    lookback_returns,
                    self.risk_free_rate,
                    self.annualization_factor
                )
                result = temp_strategy.calculate_weights(method)
                new_weights = result.weights

                # Apply volatility targeting if specified
                if target_volatility is not None:
                    portfolio_vol = result.portfolio_volatility
                    if portfolio_vol > 0:
                        leverage = min(target_volatility / portfolio_vol, max_leverage)
                        new_weights = new_weights * leverage

                # Calculate turnover
                turnover = np.sum(np.abs(new_weights - current_weights))
                turnover_history.append(turnover)

                # Apply transaction costs
                tc = turnover * transaction_cost

                current_weights = new_weights
                rebalance_dates.append(date)
                weights_history.append(current_weights.copy())
            else:
                tc = 0

            # Calculate portfolio return
            port_return = np.dot(current_weights, day_returns) - tc
            portfolio_returns.append(port_return)

            # Drift weights (no rebalance)
            if np.sum(current_weights) > 0:
                current_weights = current_weights * (1 + day_returns)
                current_weights = current_weights / np.sum(current_weights)

        # Calculate metrics
        portfolio_returns = np.array(portfolio_returns)
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(portfolio_returns)) - 1
        annualized_vol = np.std(portfolio_returns) * np.sqrt(252)
        sharpe = (annualized_return - self.risk_free_rate) / annualized_vol if annualized_vol > 0 else 0

        # Max drawdown
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = np.max(drawdowns)

        return {
            'returns': portfolio_returns,
            'dates': dates[lookback_window:],
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_turnover': np.mean(turnover_history),
            'n_rebalances': len(rebalance_dates),
            'weights_history': weights_history,
            'rebalance_dates': rebalance_dates
        }

    def run_backtest(
        self,
        initial_capital: float = 100000.0,
        **kwargs
    ) -> Dict:
        """
        Convenience wrapper for backtesting.

        Args:
            initial_capital: Starting capital
            **kwargs: Arguments for backtest method

        Returns:
            Dictionary with detailed backtest results
        """
        results = self.backtest(**kwargs)

        # Add capital tracking
        capital = initial_capital * np.cumprod(1 + results['returns'])

        results['initial_capital'] = initial_capital
        results['final_capital'] = capital[-1]
        results['capital_curve'] = capital
        results['total_return_pct'] = (capital[-1] / initial_capital - 1) * 100

        return results


def create_risk_parity_strategy(
    returns: pd.DataFrame,
    method: str = 'erc',
    target_volatility: Optional[float] = None
) -> Tuple[np.ndarray, RiskParityResult]:
    """
    Convenience function to create risk parity allocation.

    Args:
        returns: DataFrame of asset returns
        method: Risk parity method ('inverse_volatility', 'erc', 'hrp')
        target_volatility: Optional volatility target

    Returns:
        Tuple of (weights, RiskParityResult)

    Example:
        >>> weights, result = create_risk_parity_strategy(returns, method='hrp')
        >>> print(f"Weights: {weights}")
    """
    strategy = RiskParityStrategy(returns)
    result = strategy.calculate_weights(method)

    weights = result.weights

    # Scale for target volatility if specified
    if target_volatility is not None:
        if result.portfolio_volatility > 0:
            leverage = target_volatility / result.portfolio_volatility
            weights = weights * leverage

    return weights, result
