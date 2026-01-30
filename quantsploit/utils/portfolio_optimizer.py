"""
Markowitz Mean-Variance Portfolio Optimization for Quantsploit

This module provides comprehensive portfolio optimization capabilities including:
- Mean-Variance Optimization (Markowitz)
- Efficient Frontier generation
- Risk Parity and Hierarchical Risk Parity (HRP)
- Advanced constraints (sector exposure, position limits, turnover)

References:
    - Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
    - Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample.
    - Maillard, S., Roncalli, T., Teiletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios.

Author: Quantsploit Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import warnings


class OptimizationObjective(Enum):
    """Optimization objectives supported by the optimizer."""
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    TARGET_RETURN = "target_return"
    RISK_PARITY = "risk_parity"
    MAX_DIVERSIFICATION = "max_diversification"


@dataclass
class PortfolioConstraints:
    """
    Portfolio constraints for optimization.

    Attributes:
        long_only: If True, all weights must be >= 0
        fully_invested: If True, weights must sum to 1
        min_weight: Minimum weight per asset (default: 0.0)
        max_weight: Maximum weight per asset (default: 1.0)
        sector_constraints: Dict mapping sector name to (min_exposure, max_exposure)
        asset_sectors: Dict mapping asset names to their sectors
        max_turnover: Maximum allowed turnover from current_weights (None = no limit)
        current_weights: Current portfolio weights for turnover calculation
        min_assets: Minimum number of assets to hold (cardinality constraint)
        max_assets: Maximum number of assets to hold (cardinality constraint)
    """
    long_only: bool = True
    fully_invested: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0
    sector_constraints: Optional[Dict[str, Tuple[float, float]]] = None
    asset_sectors: Optional[Dict[str, str]] = None
    max_turnover: Optional[float] = None
    current_weights: Optional[np.ndarray] = None
    min_assets: Optional[int] = None
    max_assets: Optional[int] = None

    def __post_init__(self):
        """Validate constraints after initialization."""
        if self.min_weight < 0 and self.long_only:
            raise ValueError("min_weight cannot be negative when long_only is True")
        if self.max_weight > 1 and self.fully_invested:
            warnings.warn("max_weight > 1 may conflict with fully_invested constraint")
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight cannot be greater than max_weight")


@dataclass
class PortfolioMetrics:
    """
    Portfolio performance and risk metrics.

    Attributes:
        weights: Optimal portfolio weights
        expected_return: Annualized expected return
        volatility: Annualized portfolio volatility (standard deviation)
        sharpe_ratio: Sharpe ratio (excess return / volatility)
        risk_contributions: Risk contribution by each asset
        marginal_risk: Marginal risk contribution by each asset
        diversification_ratio: Diversification ratio (weighted avg vol / portfolio vol)
        effective_n: Effective number of assets (1 / sum(w^2))
        max_drawdown_estimate: Estimated max drawdown based on volatility
    """
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    risk_contributions: np.ndarray = field(default_factory=lambda: np.array([]))
    marginal_risk: np.ndarray = field(default_factory=lambda: np.array([]))
    diversification_ratio: float = 0.0
    effective_n: float = 0.0
    max_drawdown_estimate: float = 0.0

    def to_dict(self, asset_names: List[str] = None) -> Dict:
        """
        Convert metrics to dictionary for display.

        Args:
            asset_names: Optional list of asset names for weight labeling

        Returns:
            Dictionary with formatted metrics
        """
        result = {
            "Expected Return": f"{self.expected_return:.2%}",
            "Volatility": f"{self.volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Diversification Ratio": f"{self.diversification_ratio:.3f}",
            "Effective N": f"{self.effective_n:.2f}",
            "Est. Max Drawdown": f"{self.max_drawdown_estimate:.2%}",
        }

        if asset_names is not None and len(asset_names) == len(self.weights):
            result["Weights"] = {name: f"{w:.4f}" for name, w in zip(asset_names, self.weights)}
            if len(self.risk_contributions) > 0:
                result["Risk Contributions"] = {
                    name: f"{rc:.4f}" for name, rc in zip(asset_names, self.risk_contributions)
                }
        else:
            result["Weights"] = self.weights.tolist()
            if len(self.risk_contributions) > 0:
                result["Risk Contributions"] = self.risk_contributions.tolist()

        return result


class MarkowitzOptimizer:
    """
    Markowitz Mean-Variance Portfolio Optimizer.

    This class implements the classic Markowitz portfolio optimization framework
    with extensions for modern portfolio management including risk parity,
    hierarchical risk parity (HRP), and various practical constraints.

    Attributes:
        returns: DataFrame of asset returns (rows=dates, cols=assets)
        expected_returns: Expected returns for each asset (annualized)
        cov_matrix: Covariance matrix of returns (annualized)
        asset_names: List of asset names
        n_assets: Number of assets
        risk_free_rate: Risk-free rate for Sharpe calculation

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Create sample returns data
        >>> returns = pd.DataFrame(
        ...     np.random.randn(252, 5) * 0.01,
        ...     columns=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
        ... )
        >>> optimizer = MarkowitzOptimizer(returns, risk_free_rate=0.02)
        >>> result = optimizer.optimize_max_sharpe()
        >>> print(f"Optimal Sharpe: {result.sharpe_ratio:.3f}")
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02,
        annualization_factor: int = 252
    ):
        """
        Initialize the Markowitz optimizer.

        Args:
            returns: DataFrame of asset returns (daily, weekly, etc.)
                    Rows are dates, columns are assets.
            expected_returns: Optional pre-computed expected returns (annualized).
                            If None, calculated from returns using mean.
            cov_matrix: Optional pre-computed covariance matrix (annualized).
                       If None, calculated from returns.
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation.
            annualization_factor: Factor to annualize returns/volatility.
                                 252 for daily, 52 for weekly, 12 for monthly.

        Raises:
            ValueError: If returns DataFrame is empty or has invalid shape.
        """
        if returns.empty:
            raise ValueError("Returns DataFrame cannot be empty")

        self.returns = returns
        self.asset_names = list(returns.columns)
        self.n_assets = len(self.asset_names)
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor

        # Calculate or use provided expected returns (annualized)
        if expected_returns is not None:
            self.expected_returns = np.array(expected_returns)
        else:
            self.expected_returns = returns.mean().values * annualization_factor

        # Calculate or use provided covariance matrix (annualized)
        if cov_matrix is not None:
            self.cov_matrix = np.array(cov_matrix)
        else:
            self.cov_matrix = returns.cov().values * annualization_factor

        # Validate dimensions
        if len(self.expected_returns) != self.n_assets:
            raise ValueError("Expected returns length must match number of assets")
        if self.cov_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError("Covariance matrix shape must match number of assets")

    def _portfolio_return(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio expected return.

        Args:
            weights: Portfolio weights array

        Returns:
            Annualized expected portfolio return
        """
        return np.dot(weights, self.expected_returns)

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility (standard deviation).

        Args:
            weights: Portfolio weights array

        Returns:
            Annualized portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def _portfolio_sharpe(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio Sharpe ratio.

        Args:
            weights: Portfolio weights array

        Returns:
            Sharpe ratio (negative for minimization)
        """
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        if vol == 0:
            return 0
        return (ret - self.risk_free_rate) / vol

    def _neg_sharpe(self, weights: np.ndarray) -> float:
        """Negative Sharpe ratio for minimization."""
        return -self._portfolio_sharpe(weights)

    def _build_constraints(
        self,
        constraints: PortfolioConstraints,
        target_return: Optional[float] = None
    ) -> Tuple[Bounds, List]:
        """
        Build scipy optimization constraints from PortfolioConstraints.

        Args:
            constraints: PortfolioConstraints object
            target_return: Optional target return constraint

        Returns:
            Tuple of (Bounds object, list of constraint dicts)
        """
        # Bounds for individual weights
        lower_bounds = np.full(self.n_assets, constraints.min_weight)
        upper_bounds = np.full(self.n_assets, constraints.max_weight)

        if constraints.long_only:
            lower_bounds = np.maximum(lower_bounds, 0.0)

        bounds = Bounds(lower_bounds, upper_bounds)

        # List of constraints for scipy
        constraint_list = []

        # Fully invested constraint (sum = 1)
        if constraints.fully_invested:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })

        # Target return constraint
        if target_return is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda w, tr=target_return: self._portfolio_return(w) - tr
            })

        # Sector constraints
        if constraints.sector_constraints and constraints.asset_sectors:
            for sector, (min_exp, max_exp) in constraints.sector_constraints.items():
                # Get indices of assets in this sector
                sector_mask = np.array([
                    constraints.asset_sectors.get(name, '') == sector
                    for name in self.asset_names
                ])

                if np.any(sector_mask):
                    # Minimum sector exposure
                    constraint_list.append({
                        'type': 'ineq',
                        'fun': lambda w, sm=sector_mask, me=min_exp: np.sum(w[sm]) - me
                    })
                    # Maximum sector exposure
                    constraint_list.append({
                        'type': 'ineq',
                        'fun': lambda w, sm=sector_mask, me=max_exp: me - np.sum(w[sm])
                    })

        # Turnover constraint
        if constraints.max_turnover is not None and constraints.current_weights is not None:
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda w, cw=constraints.current_weights, mt=constraints.max_turnover: (
                    mt - np.sum(np.abs(w - cw))
                )
            })

        return bounds, constraint_list

    def calculate_portfolio_metrics(
        self,
        weights: np.ndarray
    ) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio metrics for given weights.

        Args:
            weights: Portfolio weight array

        Returns:
            PortfolioMetrics object with all calculated metrics
        """
        weights = np.array(weights)

        # Basic metrics
        expected_return = self._portfolio_return(weights)
        volatility = self._portfolio_volatility(weights)
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Risk contributions
        risk_contrib, marginal_risk = self.calculate_risk_contribution(weights)

        # Diversification ratio
        weighted_avg_vol = np.dot(weights, np.sqrt(np.diag(self.cov_matrix)))
        div_ratio = weighted_avg_vol / volatility if volatility > 0 else 1

        # Effective N (inverse Herfindahl)
        effective_n = 1 / np.sum(weights ** 2) if np.any(weights > 0) else 0

        # Estimated max drawdown (approximation based on volatility)
        # Using the approximation: E[MDD] ~ sqrt(2 * ln(T)) * sigma
        # Assuming T = 252 trading days
        max_dd_estimate = np.sqrt(2 * np.log(252)) * volatility

        return PortfolioMetrics(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            risk_contributions=risk_contrib,
            marginal_risk=marginal_risk,
            diversification_ratio=div_ratio,
            effective_n=effective_n,
            max_drawdown_estimate=max_dd_estimate
        )

    def calculate_risk_contribution(
        self,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate risk contribution by each asset.

        Risk contribution measures how much each asset contributes to total
        portfolio risk. The sum of risk contributions equals portfolio variance.

        Args:
            weights: Portfolio weight array

        Returns:
            Tuple of (risk_contributions, marginal_risk)
            - risk_contributions: Percentage contribution to total risk by each asset
            - marginal_risk: Marginal risk contribution (dSigma/dw * w)
        """
        weights = np.array(weights)
        portfolio_vol = self._portfolio_volatility(weights)

        if portfolio_vol == 0:
            return np.zeros(self.n_assets), np.zeros(self.n_assets)

        # Marginal contribution to risk (MCR) = Cov * w / portfolio_vol
        marginal_risk = np.dot(self.cov_matrix, weights) / portfolio_vol

        # Risk contribution = w * MCR / portfolio_vol
        risk_contrib = weights * marginal_risk / portfolio_vol

        # Normalize to get percentage contributions
        risk_contrib_pct = risk_contrib / np.sum(risk_contrib) if np.sum(risk_contrib) > 0 else risk_contrib

        return risk_contrib_pct, marginal_risk

    def optimize_min_variance(
        self,
        constraints: Optional[PortfolioConstraints] = None
    ) -> PortfolioMetrics:
        """
        Find the minimum variance portfolio.

        This optimization finds the portfolio with the lowest possible volatility
        given the specified constraints.

        Args:
            constraints: PortfolioConstraints object (uses defaults if None)

        Returns:
            PortfolioMetrics for the minimum variance portfolio

        Example:
            >>> optimizer = MarkowitzOptimizer(returns)
            >>> min_var = optimizer.optimize_min_variance()
            >>> print(f"Min Variance Vol: {min_var.volatility:.2%}")
        """
        if constraints is None:
            constraints = PortfolioConstraints()

        bounds, constraint_list = self._build_constraints(constraints)

        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            fun=lambda w: self._portfolio_volatility(w) ** 2,  # Minimize variance
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        return self.calculate_portfolio_metrics(result.x)

    def optimize_max_sharpe(
        self,
        constraints: Optional[PortfolioConstraints] = None
    ) -> PortfolioMetrics:
        """
        Find the maximum Sharpe ratio portfolio (tangent portfolio).

        This optimization finds the portfolio with the highest risk-adjusted
        return given the specified constraints.

        Args:
            constraints: PortfolioConstraints object (uses defaults if None)

        Returns:
            PortfolioMetrics for the maximum Sharpe portfolio

        Example:
            >>> optimizer = MarkowitzOptimizer(returns, risk_free_rate=0.02)
            >>> max_sharpe = optimizer.optimize_max_sharpe()
            >>> print(f"Max Sharpe Ratio: {max_sharpe.sharpe_ratio:.3f}")
        """
        if constraints is None:
            constraints = PortfolioConstraints()

        bounds, constraint_list = self._build_constraints(constraints)

        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize (minimize negative Sharpe)
        result = minimize(
            fun=self._neg_sharpe,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        return self.calculate_portfolio_metrics(result.x)

    def optimize_target_return(
        self,
        target_return: float,
        constraints: Optional[PortfolioConstraints] = None
    ) -> PortfolioMetrics:
        """
        Find the minimum variance portfolio for a target return.

        This finds a point on the efficient frontier by minimizing variance
        subject to achieving a specific expected return.

        Args:
            target_return: Target annualized return (e.g., 0.10 for 10%)
            constraints: PortfolioConstraints object (uses defaults if None)

        Returns:
            PortfolioMetrics for the portfolio with minimum variance at target return

        Raises:
            ValueError: If target return is not achievable

        Example:
            >>> optimizer = MarkowitzOptimizer(returns)
            >>> target = optimizer.optimize_target_return(0.15)  # 15% return
            >>> print(f"Volatility at 15% target: {target.volatility:.2%}")
        """
        if constraints is None:
            constraints = PortfolioConstraints()

        # Check if target is achievable
        max_possible_return = np.max(self.expected_returns)
        min_possible_return = np.min(self.expected_returns)

        if target_return > max_possible_return:
            warnings.warn(
                f"Target return {target_return:.2%} exceeds max possible {max_possible_return:.2%}"
            )

        bounds, constraint_list = self._build_constraints(constraints, target_return=target_return)

        # Initial guess: equal weights
        x0 = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            fun=lambda w: self._portfolio_volatility(w) ** 2,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        return self.calculate_portfolio_metrics(result.x)

    def efficient_frontier(
        self,
        n_points: int = 50,
        constraints: Optional[PortfolioConstraints] = None,
        include_min_variance: bool = True,
        include_max_sharpe: bool = True
    ) -> pd.DataFrame:
        """
        Generate the efficient frontier.

        Creates a set of portfolios representing the efficient frontier,
        which shows the best possible return for each level of risk.

        Args:
            n_points: Number of points on the frontier (default: 50)
            constraints: PortfolioConstraints object (uses defaults if None)
            include_min_variance: Include the minimum variance portfolio
            include_max_sharpe: Include the maximum Sharpe portfolio

        Returns:
            DataFrame with columns: return, volatility, sharpe, weights

        Example:
            >>> optimizer = MarkowitzOptimizer(returns)
            >>> frontier = optimizer.efficient_frontier(n_points=100)
            >>> frontier.plot(x='volatility', y='return')
        """
        if constraints is None:
            constraints = PortfolioConstraints()

        # Find min and max return portfolios to set range
        min_var_port = self.optimize_min_variance(constraints)

        # Find approximate max return (might be constrained)
        bounds, constraint_list = self._build_constraints(constraints)
        x0 = np.ones(self.n_assets) / self.n_assets

        max_ret_result = minimize(
            fun=lambda w: -self._portfolio_return(w),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list
        )
        max_return = -max_ret_result.fun

        # Generate target returns
        min_return = min_var_port.expected_return
        target_returns = np.linspace(min_return, max_return * 0.99, n_points)

        # Generate frontier points
        frontier_data = []

        for target in target_returns:
            try:
                metrics = self.optimize_target_return(target, constraints)
                frontier_data.append({
                    'return': metrics.expected_return,
                    'volatility': metrics.volatility,
                    'sharpe': metrics.sharpe_ratio,
                    'weights': metrics.weights.tolist(),
                    'type': 'frontier'
                })
            except Exception:
                continue

        # Add special portfolios
        if include_min_variance:
            frontier_data.append({
                'return': min_var_port.expected_return,
                'volatility': min_var_port.volatility,
                'sharpe': min_var_port.sharpe_ratio,
                'weights': min_var_port.weights.tolist(),
                'type': 'min_variance'
            })

        if include_max_sharpe:
            max_sharpe_port = self.optimize_max_sharpe(constraints)
            frontier_data.append({
                'return': max_sharpe_port.expected_return,
                'volatility': max_sharpe_port.volatility,
                'sharpe': max_sharpe_port.sharpe_ratio,
                'weights': max_sharpe_port.weights.tolist(),
                'type': 'max_sharpe'
            })

        df = pd.DataFrame(frontier_data)
        df = df.sort_values('volatility').reset_index(drop=True)

        return df

    def get_optimal_weights(
        self,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        constraints: Optional[PortfolioConstraints] = None,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Main interface for portfolio optimization.

        This is the primary method for getting optimal portfolio weights.
        It supports multiple optimization objectives and returns weights
        as a dictionary mapping asset names to weights.

        Args:
            objective: OptimizationObjective enum specifying the goal
            constraints: PortfolioConstraints object (uses defaults if None)
            target_return: Required if objective is TARGET_RETURN
            target_risk: Optional target volatility (finds closest achievable)

        Returns:
            Dictionary mapping asset names to optimal weights

        Example:
            >>> optimizer = MarkowitzOptimizer(returns)
            >>> weights = optimizer.get_optimal_weights(
            ...     objective=OptimizationObjective.MAX_SHARPE
            ... )
            >>> for asset, weight in weights.items():
            ...     print(f"{asset}: {weight:.2%}")
        """
        if objective == OptimizationObjective.MIN_VARIANCE:
            metrics = self.optimize_min_variance(constraints)
        elif objective == OptimizationObjective.MAX_SHARPE:
            metrics = self.optimize_max_sharpe(constraints)
        elif objective == OptimizationObjective.TARGET_RETURN:
            if target_return is None:
                raise ValueError("target_return required for TARGET_RETURN objective")
            metrics = self.optimize_target_return(target_return, constraints)
        elif objective == OptimizationObjective.RISK_PARITY:
            weights = self.risk_parity_weights(constraints)
            return dict(zip(self.asset_names, weights))
        elif objective == OptimizationObjective.MAX_DIVERSIFICATION:
            metrics = self._optimize_max_diversification(constraints)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # If target risk specified, find closest portfolio on frontier
        if target_risk is not None:
            frontier = self.efficient_frontier(n_points=100, constraints=constraints)
            closest_idx = (frontier['volatility'] - target_risk).abs().idxmin()
            weights = np.array(frontier.loc[closest_idx, 'weights'])
            return dict(zip(self.asset_names, weights))

        return dict(zip(self.asset_names, metrics.weights))

    def _optimize_max_diversification(
        self,
        constraints: Optional[PortfolioConstraints] = None
    ) -> PortfolioMetrics:
        """
        Maximize the diversification ratio.

        The diversification ratio is defined as the weighted average of individual
        volatilities divided by portfolio volatility.
        """
        if constraints is None:
            constraints = PortfolioConstraints()

        individual_vols = np.sqrt(np.diag(self.cov_matrix))

        def neg_div_ratio(weights):
            weighted_avg_vol = np.dot(weights, individual_vols)
            portfolio_vol = self._portfolio_volatility(weights)
            return -weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0

        bounds, constraint_list = self._build_constraints(constraints)
        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            fun=neg_div_ratio,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )

        return self.calculate_portfolio_metrics(result.x)

    def risk_parity_weights(
        self,
        constraints: Optional[PortfolioConstraints] = None,
        risk_budget: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate risk parity (equal risk contribution) weights.

        Risk parity allocates weights such that each asset contributes equally
        to total portfolio risk. This approach avoids concentration in
        high-volatility assets that often occurs with mean-variance optimization.

        Args:
            constraints: PortfolioConstraints object (uses defaults if None)
            risk_budget: Optional target risk budget per asset.
                        If None, equal risk budget (1/n) is used.

        Returns:
            Array of optimal weights achieving risk parity

        Example:
            >>> optimizer = MarkowitzOptimizer(returns)
            >>> rp_weights = optimizer.risk_parity_weights()
            >>> print("Risk Parity Weights:", dict(zip(optimizer.asset_names, rp_weights)))

        References:
            Maillard, S., Roncalli, T., Teiletche, J. (2010).
            "The Properties of Equally Weighted Risk Contribution Portfolios"
        """
        if constraints is None:
            constraints = PortfolioConstraints()

        if risk_budget is None:
            risk_budget = np.ones(self.n_assets) / self.n_assets
        else:
            risk_budget = np.array(risk_budget)
            risk_budget = risk_budget / np.sum(risk_budget)  # Normalize

        def risk_parity_objective(weights):
            """Objective: minimize deviation from target risk contributions."""
            portfolio_vol = self._portfolio_volatility(weights)
            if portfolio_vol == 0:
                return 1e10

            # Risk contributions
            marginal_contrib = np.dot(self.cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            risk_contrib_pct = risk_contrib / np.sum(risk_contrib)

            # Sum of squared deviations from target
            return np.sum((risk_contrib_pct - risk_budget) ** 2)

        bounds, constraint_list = self._build_constraints(constraints)

        # Better initial guess: inverse volatility weighted
        initial_weights = 1 / np.sqrt(np.diag(self.cov_matrix))
        initial_weights = initial_weights / np.sum(initial_weights)

        result = minimize(
            fun=risk_parity_objective,
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )

        if not result.success:
            warnings.warn(f"Risk parity optimization did not converge: {result.message}")

        return result.x

    def hierarchical_risk_parity(
        self,
        linkage_method: str = 'ward'
    ) -> np.ndarray:
        """
        Hierarchical Risk Parity (HRP) method by Lopez de Prado.

        HRP is a machine learning approach to portfolio optimization that:
        1. Uses hierarchical clustering to group correlated assets
        2. Applies recursive bisection to allocate weights
        3. Does not require covariance matrix inversion (more stable)

        This method often outperforms traditional mean-variance optimization
        out-of-sample due to better handling of estimation error.

        Args:
            linkage_method: Hierarchical clustering method.
                          Options: 'single', 'complete', 'average', 'ward'
                          Default: 'ward' (minimizes variance)

        Returns:
            Array of HRP weights

        Example:
            >>> optimizer = MarkowitzOptimizer(returns)
            >>> hrp_weights = optimizer.hierarchical_risk_parity()
            >>> print("HRP Weights:", dict(zip(optimizer.asset_names, hrp_weights)))

        References:
            Lopez de Prado, M. (2016). "Building Diversified Portfolios
            that Outperform Out-of-Sample". Journal of Portfolio Management.
        """
        # Step 1: Calculate correlation and distance matrix
        corr = self.returns.corr().values

        # Ensure correlation matrix is valid
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)

        # Distance matrix: d = sqrt(0.5 * (1 - corr))
        dist = np.sqrt(0.5 * (1 - corr))

        # Step 2: Hierarchical clustering
        dist_condensed = squareform(dist)
        link = linkage(dist_condensed, method=linkage_method)

        # Get sorted indices from clustering
        sorted_idx = leaves_list(link)

        # Step 3: Recursive bisection with inverse variance allocation
        weights = self._recursive_bisection(sorted_idx)

        return weights

    def _recursive_bisection(self, sorted_idx: np.ndarray) -> np.ndarray:
        """
        Perform recursive bisection for HRP.

        Args:
            sorted_idx: Sorted asset indices from clustering

        Returns:
            Array of weights
        """
        weights = np.ones(self.n_assets)

        # Cluster weights (initially all in one cluster)
        cluster_items = [sorted_idx.tolist()]

        while cluster_items:
            # Split each cluster
            new_clusters = []

            for cluster in cluster_items:
                if len(cluster) == 1:
                    continue

                # Split in half
                mid = len(cluster) // 2
                left_cluster = cluster[:mid]
                right_cluster = cluster[mid:]

                # Calculate cluster variances
                left_var = self._get_cluster_variance(left_cluster)
                right_var = self._get_cluster_variance(right_cluster)

                # Inverse variance allocation
                total_inv_var = 1 / left_var + 1 / right_var
                left_weight = (1 / left_var) / total_inv_var
                right_weight = (1 / right_var) / total_inv_var

                # Apply weights
                weights[left_cluster] *= left_weight
                weights[right_cluster] *= right_weight

                # Add sub-clusters for next iteration
                if len(left_cluster) > 1:
                    new_clusters.append(left_cluster)
                if len(right_cluster) > 1:
                    new_clusters.append(right_cluster)

            cluster_items = new_clusters

        return weights

    def _get_cluster_variance(self, indices: List[int]) -> float:
        """
        Calculate variance of a cluster using inverse-variance weighted portfolio.

        Args:
            indices: List of asset indices in the cluster

        Returns:
            Cluster variance
        """
        if len(indices) == 1:
            return self.cov_matrix[indices[0], indices[0]]

        # Get sub-covariance matrix
        sub_cov = self.cov_matrix[np.ix_(indices, indices)]

        # Inverse variance weights within cluster
        inv_diag = 1 / np.diag(sub_cov)
        inv_var_weights = inv_diag / np.sum(inv_diag)

        # Calculate cluster variance
        return np.dot(inv_var_weights.T, np.dot(sub_cov, inv_var_weights))

    # =========================================================================
    # RISK PARITY ENHANCEMENTS (Added 2026-01-23)
    # =========================================================================

    def risk_parity_targeted(
        self,
        target_volatility: float = 0.10,
        constraints: Optional[PortfolioConstraints] = None,
        risk_budget: Optional[np.ndarray] = None,
        use_leverage: bool = False,
        max_leverage: float = 3.0
    ) -> Dict:
        """
        Risk Parity with volatility targeting.

        Calculates risk parity weights and optionally applies leverage to achieve
        a target portfolio volatility. Essential for comparing risk parity
        strategies on an equal-risk basis with other approaches.

        Args:
            target_volatility: Target annualized portfolio volatility (e.g., 0.10 for 10%)
            constraints: PortfolioConstraints object (uses defaults if None)
            risk_budget: Optional custom risk budget per asset
            use_leverage: If True, apply leverage to hit target volatility
            max_leverage: Maximum allowed leverage multiplier

        Returns:
            Dictionary with:
                - weights: Raw risk parity weights (sum to 1)
                - leveraged_weights: Weights after leverage (may sum > 1)
                - leverage: Leverage multiplier applied
                - target_vol: Target volatility requested
                - achieved_vol: Actual portfolio volatility
                - expected_return: Expected portfolio return (before leverage cost)

        Example:
            >>> optimizer = MarkowitzOptimizer(returns)
            >>> result = optimizer.risk_parity_targeted(target_volatility=0.12)
            >>> print(f"Leverage: {result['leverage']:.2f}x")
            >>> print(f"Achieved Vol: {result['achieved_vol']:.2%}")

        References:
            - Bridgewater Associates "All Weather" Strategy
            - Risk Parity Fundamentals, Qian (2005)
        """
        # Get base risk parity weights
        base_weights = self.risk_parity_weights(constraints, risk_budget)

        # Calculate base portfolio volatility
        base_vol = self._portfolio_volatility(base_weights)
        base_return = self._portfolio_return(base_weights)

        if base_vol == 0:
            warnings.warn("Base portfolio volatility is zero, cannot target volatility")
            return {
                'weights': base_weights,
                'leveraged_weights': base_weights,
                'leverage': 1.0,
                'target_vol': target_volatility,
                'achieved_vol': 0.0,
                'expected_return': base_return
            }

        # Calculate required leverage
        required_leverage = target_volatility / base_vol

        if use_leverage:
            # Apply leverage with cap
            leverage = min(required_leverage, max_leverage)
            leveraged_weights = base_weights * leverage
            achieved_vol = base_vol * leverage
            leveraged_return = base_return * leverage
        else:
            # No leverage - just report what we'd need
            leverage = 1.0
            leveraged_weights = base_weights
            achieved_vol = base_vol
            leveraged_return = base_return

        return {
            'weights': base_weights,
            'leveraged_weights': leveraged_weights,
            'leverage': leverage,
            'required_leverage': required_leverage,
            'target_vol': target_volatility,
            'achieved_vol': achieved_vol,
            'expected_return': leveraged_return,
            'base_vol': base_vol,
            'base_return': base_return
        }

    def leveraged_risk_parity(
        self,
        target_return: Optional[float] = None,
        target_volatility: float = 0.10,
        max_leverage: float = 2.5,
        leverage_cost: float = 0.02,
        constraints: Optional[PortfolioConstraints] = None,
        risk_budget: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Bridgewater-style Leveraged Risk Parity.

        Implements leveraged risk parity similar to Bridgewater's All Weather
        strategy. Uses leverage on low-volatility assets (typically bonds) to
        achieve comparable risk contribution to higher-volatility assets (equities).

        Args:
            target_return: Optional target return (leverage solved to achieve)
            target_volatility: Target portfolio volatility if no target_return
            max_leverage: Maximum allowed leverage (typical range: 1.5-3.0)
            leverage_cost: Annual cost of borrowing (e.g., 0.02 for 2%)
            constraints: PortfolioConstraints object
            risk_budget: Optional custom risk budget

        Returns:
            Dictionary with:
                - weights: Base risk parity weights
                - leveraged_weights: Weights after leverage
                - leverage: Applied leverage multiplier
                - gross_exposure: Total absolute weight (100% = no leverage)
                - expected_return: Gross expected return
                - net_expected_return: Return after leverage costs
                - expected_vol: Portfolio volatility
                - sharpe_ratio: Net Sharpe ratio
                - cost_of_leverage: Annual cost of leverage

        Example:
            >>> optimizer = MarkowitzOptimizer(returns)
            >>> result = optimizer.leveraged_risk_parity(
            ...     target_volatility=0.12,
            ...     max_leverage=2.0,
            ...     leverage_cost=0.03
            ... )
            >>> print(f"Net Return: {result['net_expected_return']:.2%}")
            >>> print(f"Leverage: {result['leverage']:.2f}x")

        References:
            - Bridgewater Associates "All Weather" white paper
            - "Risk Parity Portfolios with Leverage" - Asness, Frazzini, Pedersen (2012)
        """
        # Get base risk parity weights
        base_weights = self.risk_parity_weights(constraints, risk_budget)

        base_vol = self._portfolio_volatility(base_weights)
        base_return = self._portfolio_return(base_weights)

        if base_vol == 0:
            warnings.warn("Base portfolio volatility is zero")
            return self._empty_leveraged_result(base_weights, leverage_cost)

        # Determine leverage
        if target_return is not None:
            # Solve for leverage to achieve target return (accounting for cost)
            # target_return = leverage * base_return - (leverage - 1) * leverage_cost
            # Rearranging: leverage = target_return / (base_return - leverage_cost) + leverage_cost / (base_return - leverage_cost)
            if base_return <= leverage_cost:
                warnings.warn("Base return <= leverage cost, cannot achieve positive net return with leverage")
                leverage = 1.0
            else:
                leverage = (target_return + leverage_cost) / (base_return)
                leverage = min(leverage, max_leverage)
        else:
            # Target volatility
            leverage = min(target_volatility / base_vol, max_leverage)

        # Apply leverage
        leveraged_weights = base_weights * leverage
        gross_exposure = np.sum(np.abs(leveraged_weights))

        # Calculate returns
        gross_return = base_return * leverage
        cost_of_leverage = (leverage - 1) * leverage_cost if leverage > 1 else 0
        net_return = gross_return - cost_of_leverage

        # Calculate risk metrics
        achieved_vol = base_vol * leverage
        sharpe = (net_return - self.risk_free_rate) / achieved_vol if achieved_vol > 0 else 0

        return {
            'weights': base_weights,
            'leveraged_weights': leveraged_weights,
            'leverage': leverage,
            'gross_exposure': gross_exposure,
            'expected_return': gross_return,
            'net_expected_return': net_return,
            'expected_vol': achieved_vol,
            'sharpe_ratio': sharpe,
            'cost_of_leverage': cost_of_leverage,
            'leverage_cost_rate': leverage_cost,
            'base_return': base_return,
            'base_vol': base_vol
        }

    def _empty_leveraged_result(self, weights: np.ndarray, leverage_cost: float) -> Dict:
        """Return empty result structure for edge cases."""
        return {
            'weights': weights,
            'leveraged_weights': weights,
            'leverage': 1.0,
            'gross_exposure': 1.0,
            'expected_return': 0.0,
            'net_expected_return': 0.0,
            'expected_vol': 0.0,
            'sharpe_ratio': 0.0,
            'cost_of_leverage': 0.0,
            'leverage_cost_rate': leverage_cost,
            'base_return': 0.0,
            'base_vol': 0.0
        }

    def risk_parity_rebalance(
        self,
        current_weights: np.ndarray,
        cost_model: Optional[Callable] = None,
        rebalance_threshold: float = 0.05,
        min_trade_size: float = 0.01,
        constraints: Optional[PortfolioConstraints] = None,
        risk_budget: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Transaction cost-aware risk parity rebalancing.

        Calculates optimal rebalancing from current weights to risk parity,
        considering transaction costs and implementing rebalancing bands
        to avoid excessive turnover.

        Args:
            current_weights: Current portfolio weights
            cost_model: Optional function(trade_size) -> cost. If None, uses
                       simple proportional cost of 0.1%
            rebalance_threshold: Only rebalance if drift exceeds this (e.g., 0.05 = 5%)
            min_trade_size: Minimum trade size to execute (avoid tiny trades)
            constraints: PortfolioConstraints object
            risk_budget: Optional custom risk budget

        Returns:
            Dictionary with:
                - target_weights: Ideal risk parity weights
                - recommended_weights: Cost-optimal weights after rebalancing
                - trades: Dict of asset -> trade_amount
                - total_turnover: Sum of absolute weight changes
                - estimated_cost: Estimated transaction costs
                - should_rebalance: Boolean indicating if rebalancing is recommended
                - drift_from_target: Max drift of any asset from target

        Example:
            >>> current = np.array([0.25, 0.30, 0.20, 0.25])
            >>> result = optimizer.risk_parity_rebalance(current)
            >>> if result['should_rebalance']:
            ...     print(f"Execute trades: {result['trades']}")
            ...     print(f"Est. cost: {result['estimated_cost']:.4%}")
        """
        current_weights = np.array(current_weights)

        # Default cost model: 10 bps proportional
        if cost_model is None:
            cost_model = lambda trade_size: abs(trade_size) * 0.001

        # Get target risk parity weights
        target_weights = self.risk_parity_weights(constraints, risk_budget)

        # Calculate drift
        weight_diff = target_weights - current_weights
        max_drift = np.max(np.abs(weight_diff))
        total_turnover = np.sum(np.abs(weight_diff))

        # Check if rebalancing is needed
        should_rebalance = max_drift > rebalance_threshold

        if not should_rebalance:
            # No rebalancing needed - stay at current weights
            return {
                'target_weights': target_weights,
                'recommended_weights': current_weights,
                'trades': {name: 0.0 for name in self.asset_names},
                'total_turnover': 0.0,
                'estimated_cost': 0.0,
                'should_rebalance': False,
                'drift_from_target': max_drift,
                'rebalance_threshold': rebalance_threshold
            }

        # Calculate optimal trades (simple approach: full rebalance to target)
        # Filter out trades below minimum size
        trades = {}
        recommended_weights = current_weights.copy()
        total_cost = 0.0

        for i, (name, diff) in enumerate(zip(self.asset_names, weight_diff)):
            if abs(diff) >= min_trade_size:
                trades[name] = float(diff)
                recommended_weights[i] = target_weights[i]
                total_cost += cost_model(diff)
            else:
                trades[name] = 0.0

        # Renormalize weights to sum to 1
        if np.sum(recommended_weights) > 0:
            recommended_weights = recommended_weights / np.sum(recommended_weights)

        actual_turnover = sum(abs(t) for t in trades.values())

        return {
            'target_weights': target_weights,
            'recommended_weights': recommended_weights,
            'trades': trades,
            'total_turnover': actual_turnover,
            'estimated_cost': total_cost,
            'should_rebalance': True,
            'drift_from_target': max_drift,
            'rebalance_threshold': rebalance_threshold
        }

    def risk_parity_garch(
        self,
        garch_cov_matrix: Optional[np.ndarray] = None,
        use_ewma_fallback: bool = True,
        ewma_lambda: float = 0.94,
        constraints: Optional[PortfolioConstraints] = None,
        risk_budget: Optional[np.ndarray] = None
    ) -> Dict:
        """
        GARCH-based Risk Parity using conditional volatility forecasts.

        Uses GARCH-forecasted covariance matrix instead of historical covariance
        for more responsive risk parity allocation that adapts to changing
        market volatility regimes.

        Args:
            garch_cov_matrix: Pre-computed GARCH covariance forecast. If None,
                            uses EWMA as fallback.
            use_ewma_fallback: If True and no GARCH cov provided, use EWMA
            ewma_lambda: EWMA decay factor (default 0.94 = RiskMetrics)
            constraints: PortfolioConstraints object
            risk_budget: Optional custom risk budget

        Returns:
            Dictionary with:
                - weights: GARCH-based risk parity weights
                - cov_matrix_used: The covariance matrix used ('garch', 'ewma', 'historical')
                - conditional_vols: Conditional volatility for each asset
                - metrics: PortfolioMetrics for the resulting portfolio

        Example:
            >>> # Using with volatility_models.py GARCH forecasts
            >>> from quantsploit.utils.volatility_models import GARCHModel
            >>> garch_cov = compute_garch_covariance(returns)  # Your implementation
            >>> result = optimizer.risk_parity_garch(garch_cov_matrix=garch_cov)

        Note:
            For full GARCH functionality, integrate with volatility_models.py.
            This method provides EWMA fallback for quick conditional volatility.
        """
        original_cov = self.cov_matrix.copy()

        if garch_cov_matrix is not None:
            # Use provided GARCH covariance
            self.cov_matrix = np.array(garch_cov_matrix)
            cov_type = 'garch'
        elif use_ewma_fallback:
            # Compute EWMA covariance
            ewma_cov = self._compute_ewma_covariance(ewma_lambda)
            self.cov_matrix = ewma_cov
            cov_type = 'ewma'
        else:
            cov_type = 'historical'

        # Calculate risk parity with conditional covariance
        weights = self.risk_parity_weights(constraints, risk_budget)
        conditional_vols = np.sqrt(np.diag(self.cov_matrix))

        # Calculate metrics
        metrics = self.calculate_portfolio_metrics(weights)

        # Restore original covariance
        self.cov_matrix = original_cov

        return {
            'weights': weights,
            'cov_matrix_used': cov_type,
            'conditional_vols': conditional_vols,
            'metrics': metrics,
            'ewma_lambda': ewma_lambda if cov_type == 'ewma' else None
        }

    def _compute_ewma_covariance(self, lambda_param: float = 0.94) -> np.ndarray:
        """
        Compute EWMA (Exponentially Weighted Moving Average) covariance matrix.

        RiskMetrics-style exponential smoothing for covariance estimation.

        Args:
            lambda_param: Decay factor (0.94 = RiskMetrics daily, 0.97 = monthly)

        Returns:
            EWMA covariance matrix (annualized)
        """
        returns_array = self.returns.values
        n_obs, n_assets = returns_array.shape

        # Initialize with sample covariance
        ewma_cov = np.cov(returns_array.T)

        # Apply EWMA recursively
        for t in range(1, n_obs):
            ret_t = returns_array[t:t+1].T  # Column vector
            outer_prod = ret_t @ ret_t.T
            ewma_cov = lambda_param * ewma_cov + (1 - lambda_param) * outer_prod

        # Annualize
        return ewma_cov * self.annualization_factor

    def hierarchical_risk_parity_constrained(
        self,
        constraints: PortfolioConstraints,
        linkage_method: str = 'ward',
        max_iterations: int = 10,
        tolerance: float = 1e-6
    ) -> np.ndarray:
        """
        Hierarchical Risk Parity with constraint support.

        Standard HRP doesn't support constraints like weight limits or sector
        exposure. This method implements constrained HRP via iterative projection.

        Algorithm:
        1. Calculate unconstrained HRP weights
        2. Project onto constraint set
        3. Use projected weights as prior for re-weighted HRP
        4. Repeat until convergence

        Args:
            constraints: PortfolioConstraints with weight limits, sectors, etc.
            linkage_method: Clustering method ('ward', 'single', 'complete', 'average')
            max_iterations: Maximum projection iterations
            tolerance: Convergence tolerance for weight changes

        Returns:
            Constrained HRP weights

        Example:
            >>> constraints = PortfolioConstraints(
            ...     max_weight=0.25,
            ...     sector_constraints={'Tech': (0.0, 0.40)}
            ... )
            >>> weights = optimizer.hierarchical_risk_parity_constrained(constraints)
        """
        # Get unconstrained HRP weights
        weights = self.hierarchical_risk_parity(linkage_method)

        for iteration in range(max_iterations):
            old_weights = weights.copy()

            # Project onto constraint set
            weights = self._project_weights_to_constraints(weights, constraints)

            # Check convergence
            if np.max(np.abs(weights - old_weights)) < tolerance:
                break

        return weights

    def _project_weights_to_constraints(
        self,
        weights: np.ndarray,
        constraints: PortfolioConstraints
    ) -> np.ndarray:
        """
        Project weights onto the constraint set.

        Uses simple iterative projection for bound constraints and
        normalization for the sum-to-one constraint.

        Args:
            weights: Input weights to project
            constraints: Constraint specification

        Returns:
            Projected weights satisfying constraints
        """
        projected = weights.copy()

        # Apply bound constraints
        if constraints.long_only:
            projected = np.maximum(projected, 0)

        projected = np.maximum(projected, constraints.min_weight)
        projected = np.minimum(projected, constraints.max_weight)

        # Apply sector constraints if specified
        if constraints.sector_constraints and constraints.asset_sectors:
            for sector, (min_exp, max_exp) in constraints.sector_constraints.items():
                sector_mask = np.array([
                    constraints.asset_sectors.get(name, '') == sector
                    for name in self.asset_names
                ])

                if np.any(sector_mask):
                    sector_weight = np.sum(projected[sector_mask])

                    if sector_weight > max_exp:
                        # Scale down sector weights
                        scale = max_exp / sector_weight
                        projected[sector_mask] *= scale
                    elif sector_weight < min_exp and sector_weight > 0:
                        # Scale up sector weights
                        scale = min_exp / sector_weight
                        projected[sector_mask] *= scale

        # Normalize to sum to 1 if fully invested
        if constraints.fully_invested and np.sum(projected) > 0:
            projected = projected / np.sum(projected)

        return projected

    # =========================================================================
    # BLACK-LITTERMAN MODEL (Added 2026-01-28)
    # =========================================================================

    def black_litterman(
        self,
        market_caps: np.ndarray,
        views: Optional[List[Dict]] = None,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        omega_method: str = 'proportional',
        constraints: Optional[PortfolioConstraints] = None
    ) -> Dict:
        """
        Black-Litterman model for incorporating investor views into portfolio optimization.

        The Black-Litterman model solves key issues with Markowitz optimization:
        1. Uses market equilibrium as a prior (CAPM implied returns)
        2. Allows explicit incorporation of investor views with uncertainty
        3. Produces more intuitive, diversified portfolios
        4. Handles estimation error better through Bayesian framework

        ★ Insight ─────────────────────────────────────
        Black-Litterman addresses the "input sensitivity" problem of MVO:
        - Instead of raw expected returns (highly error-prone), it starts
          with market-implied returns derived from equilibrium
        - Investor views are blended using Bayesian updating
        - The result is more stable, intuitive allocations
        ─────────────────────────────────────────────────

        Args:
            market_caps: Market capitalizations for each asset (used to derive
                        equilibrium weights and implied returns)
            views: List of view dictionaries, each containing:
                - 'assets': List of asset indices involved in view
                - 'weights': View weights (e.g., [1, -1] for relative view)
                - 'return': Expected return of the view (e.g., 0.02 for 2%)
                - 'confidence': Confidence in view (0-1, default 1.0)
            tau: Scalar indicating uncertainty in equilibrium (typically 0.01-0.10)
            risk_aversion: Market risk aversion parameter (typically 2-3)
            omega_method: Method for setting view uncertainty:
                - 'proportional': Omega = tau * P * Sigma * P' (He-Litterman)
                - 'confidence': Use view confidence to scale omega
                - 'idzorek': Idzorek's method for intuitive confidence
            constraints: Portfolio constraints for final optimization

        Returns:
            Dictionary with:
                - weights: Optimal Black-Litterman weights
                - expected_returns: Posterior expected returns
                - equilibrium_returns: CAPM implied equilibrium returns (Pi)
                - prior_cov: Prior covariance (tau * Sigma)
                - posterior_cov: Posterior covariance matrix
                - metrics: Portfolio metrics for the optimal portfolio

        Example:
            >>> # Absolute view: Tech will return 15% next year
            >>> # Relative view: Financials will outperform Energy by 3%
            >>> market_caps = np.array([1e12, 500e9, 300e9, 200e9, 150e9])
            >>> views = [
            ...     {'assets': [0], 'weights': [1], 'return': 0.15, 'confidence': 0.7},
            ...     {'assets': [1, 2], 'weights': [1, -1], 'return': 0.03, 'confidence': 0.5}
            ... ]
            >>> result = optimizer.black_litterman(market_caps, views)
            >>> print(f"BL Weights: {result['weights']}")

        References:
            - Black, F. and Litterman, R. (1992). "Global Portfolio Optimization"
            - He, G. and Litterman, R. (1999). "The Intuition Behind Black-Litterman"
            - Idzorek, T. (2005). "A Step-by-Step Guide to the Black-Litterman Model"
        """
        market_caps = np.array(market_caps)

        # Step 1: Calculate equilibrium weights from market caps
        equilibrium_weights = market_caps / np.sum(market_caps)

        # Step 2: Calculate implied equilibrium returns (reverse optimization)
        # Pi = delta * Sigma * w_mkt
        # where delta = risk aversion, Sigma = covariance, w_mkt = market weights
        equilibrium_returns = risk_aversion * self.cov_matrix @ equilibrium_weights

        # Step 3: Prior covariance for returns
        prior_cov = tau * self.cov_matrix

        # If no views provided, use equilibrium
        if views is None or len(views) == 0:
            posterior_returns = equilibrium_returns
            posterior_cov = prior_cov
        else:
            # Step 4: Construct view matrices P and Q
            n_views = len(views)
            P = np.zeros((n_views, self.n_assets))
            Q = np.zeros(n_views)
            confidences = np.ones(n_views)

            for i, view in enumerate(views):
                for j, (asset_idx, weight) in enumerate(zip(view['assets'], view['weights'])):
                    P[i, asset_idx] = weight
                Q[i] = view['return']
                confidences[i] = view.get('confidence', 1.0)

            # Step 5: Construct Omega (view uncertainty matrix)
            if omega_method == 'proportional':
                # He-Litterman: Omega = tau * P * Sigma * P'
                # This makes view uncertainty proportional to asset variance
                Omega = tau * P @ self.cov_matrix @ P.T
            elif omega_method == 'confidence':
                # Scale diagonal by inverse confidence
                base_omega = tau * P @ self.cov_matrix @ P.T
                confidence_scale = 1 / (confidences ** 2)
                Omega = base_omega * np.diag(confidence_scale)
            elif omega_method == 'idzorek':
                # Idzorek method: confidence maps to "tilt" between prior and 100% view
                Omega = self._idzorek_omega(P, Q, equilibrium_returns, confidences)
            else:
                raise ValueError(f"Unknown omega_method: {omega_method}")

            # Step 6: Black-Litterman formula for posterior
            # E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
            prior_precision = np.linalg.inv(prior_cov)
            omega_inv = np.linalg.inv(Omega)

            # Posterior precision
            posterior_precision = prior_precision + P.T @ omega_inv @ P

            # Posterior covariance
            posterior_cov = np.linalg.inv(posterior_precision)

            # Posterior mean (expected returns)
            posterior_returns = posterior_cov @ (prior_precision @ equilibrium_returns + P.T @ omega_inv @ Q)

        # Step 7: Optimize with posterior returns
        # Store original expected returns
        original_returns = self.expected_returns.copy()

        # Use posterior returns for optimization
        self.expected_returns = posterior_returns

        if constraints is None:
            constraints = PortfolioConstraints()

        # Optimize using maximum Sharpe with posterior parameters
        optimal_portfolio = self.optimize_max_sharpe(constraints)

        # Restore original expected returns
        self.expected_returns = original_returns

        return {
            'weights': optimal_portfolio.weights,
            'expected_returns': posterior_returns,
            'equilibrium_returns': equilibrium_returns,
            'equilibrium_weights': equilibrium_weights,
            'prior_cov': prior_cov,
            'posterior_cov': posterior_cov,
            'tau': tau,
            'risk_aversion': risk_aversion,
            'metrics': optimal_portfolio
        }

    def _idzorek_omega(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        equilibrium_returns: np.ndarray,
        confidences: np.ndarray
    ) -> np.ndarray:
        """
        Idzorek's intuitive method for specifying view uncertainty.

        Maps confidence (0-1) to implied omega such that:
        - Confidence 0: Weight stays at equilibrium
        - Confidence 1: Full tilt to view

        Args:
            P: View matrix (n_views x n_assets)
            Q: View return vector
            equilibrium_returns: Prior equilibrium returns
            confidences: Confidence levels for each view (0-1)

        Returns:
            Omega diagonal matrix
        """
        n_views = len(Q)
        omega_diag = np.zeros(n_views)

        for i in range(n_views):
            # View return from equilibrium
            eq_view_return = P[i] @ equilibrium_returns

            # Target posterior weight deviation
            # Higher confidence -> smaller omega -> more weight on view
            if confidences[i] >= 1.0:
                omega_diag[i] = 1e-10  # Near-certain view
            elif confidences[i] <= 0.0:
                omega_diag[i] = 1e10  # No confidence, stay at prior
            else:
                # Calibrate omega so that posterior is confidence-weighted blend
                # This is an approximation of Idzorek's iterative method
                omega_diag[i] = (P[i] @ self.cov_matrix @ P[i].T) * ((1 - confidences[i]) / confidences[i])

        return np.diag(omega_diag)

    def black_litterman_views_from_forecasts(
        self,
        return_forecasts: Dict[str, float],
        forecast_confidence: float = 0.5,
        relative_views: Optional[List[Tuple[str, str, float]]] = None,
        relative_confidence: float = 0.6
    ) -> List[Dict]:
        """
        Helper to construct Black-Litterman views from return forecasts.

        Converts analyst forecasts or model predictions into properly
        formatted Black-Litterman views.

        Args:
            return_forecasts: Dict mapping asset names to expected returns
                            (e.g., {'AAPL': 0.12, 'GOOGL': 0.10})
            forecast_confidence: Default confidence for absolute forecasts
            relative_views: List of relative views as tuples:
                          (outperformer, underperformer, excess_return)
                          e.g., [('AAPL', 'GOOGL', 0.02)]  # AAPL beats GOOGL by 2%
            relative_confidence: Default confidence for relative views

        Returns:
            List of view dictionaries ready for black_litterman()

        Example:
            >>> forecasts = {'AAPL': 0.15, 'MSFT': 0.12, 'GOOGL': 0.10}
            >>> relative = [('AAPL', 'GOOGL', 0.05)]  # AAPL +5% vs GOOGL
            >>> views = optimizer.black_litterman_views_from_forecasts(
            ...     forecasts, relative_views=relative
            ... )
            >>> result = optimizer.black_litterman(market_caps, views)
        """
        views = []

        # Convert absolute forecasts to views
        for asset_name, expected_return in return_forecasts.items():
            if asset_name in self.asset_names:
                asset_idx = self.asset_names.index(asset_name)
                views.append({
                    'assets': [asset_idx],
                    'weights': [1.0],
                    'return': expected_return,
                    'confidence': forecast_confidence
                })

        # Add relative views
        if relative_views is not None:
            for outperformer, underperformer, excess_return in relative_views:
                if outperformer in self.asset_names and underperformer in self.asset_names:
                    out_idx = self.asset_names.index(outperformer)
                    under_idx = self.asset_names.index(underperformer)
                    views.append({
                        'assets': [out_idx, under_idx],
                        'weights': [1.0, -1.0],
                        'return': excess_return,
                        'confidence': relative_confidence
                    })

        return views

    def dynamic_risk_budget(
        self,
        regime: str = 'neutral',
        regime_budgets: Optional[Dict[str, np.ndarray]] = None,
        constraints: Optional[PortfolioConstraints] = None
    ) -> Dict:
        """
        Risk Parity with regime-dependent risk budgets.

        Adjusts risk budgets based on market regime (volatility environment).
        In high-volatility regimes, may shift risk budget toward defensive assets.

        Args:
            regime: Current regime ('low_vol', 'neutral', 'high_vol', 'crisis')
            regime_budgets: Dict mapping regime names to risk budget arrays.
                          If None, uses default budgets.
            constraints: PortfolioConstraints object

        Returns:
            Dictionary with:
                - weights: Risk parity weights for current regime
                - risk_budget: The risk budget used
                - regime: The regime applied
                - metrics: Portfolio metrics

        Example:
            >>> # Define regime-specific risk budgets
            >>> budgets = {
            ...     'low_vol': np.array([0.4, 0.3, 0.3]),    # More risk to growth
            ...     'high_vol': np.array([0.2, 0.4, 0.4]),   # More to defensive
            ...     'neutral': np.array([0.33, 0.33, 0.34])  # Equal risk
            ... }
            >>> result = optimizer.dynamic_risk_budget(regime='high_vol', regime_budgets=budgets)

        Note:
            Integrate with VolatilityRegimeDetector from volatility_models.py
            for automatic regime detection.
        """
        # Default regime budgets (equal)
        default_budget = np.ones(self.n_assets) / self.n_assets

        if regime_budgets is None:
            regime_budgets = {
                'low_vol': default_budget,
                'neutral': default_budget,
                'high_vol': default_budget,
                'crisis': default_budget
            }

        # Get budget for current regime
        risk_budget = regime_budgets.get(regime, default_budget)
        risk_budget = np.array(risk_budget)

        # Validate and normalize
        if len(risk_budget) != self.n_assets:
            warnings.warn(f"Risk budget length mismatch, using equal budget")
            risk_budget = default_budget
        risk_budget = risk_budget / np.sum(risk_budget)

        # Calculate risk parity with regime-adjusted budget
        weights = self.risk_parity_weights(constraints, risk_budget)
        metrics = self.calculate_portfolio_metrics(weights)

        return {
            'weights': weights,
            'risk_budget': risk_budget,
            'regime': regime,
            'metrics': metrics
        }

    def plot_efficient_frontier(
        self,
        n_points: int = 50,
        constraints: Optional[PortfolioConstraints] = None,
        show_assets: bool = True,
        show_special_portfolios: bool = True,
        ax=None
    ):
        """
        Plot the efficient frontier with optional asset positions.

        Creates a visualization of the efficient frontier showing the
        risk-return tradeoff of optimal portfolios.

        Args:
            n_points: Number of frontier points to calculate
            constraints: PortfolioConstraints object (uses defaults if None)
            show_assets: If True, plot individual asset positions
            show_special_portfolios: If True, highlight min var and max Sharpe
            ax: Matplotlib axes object (creates new figure if None)

        Returns:
            Matplotlib axes object

        Example:
            >>> optimizer = MarkowitzOptimizer(returns)
            >>> ax = optimizer.plot_efficient_frontier()
            >>> plt.show()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Generate frontier
        frontier = self.efficient_frontier(n_points, constraints)
        frontier_main = frontier[frontier['type'] == 'frontier']

        # Plot efficient frontier
        ax.plot(
            frontier_main['volatility'] * 100,
            frontier_main['return'] * 100,
            'b-',
            linewidth=2,
            label='Efficient Frontier'
        )

        # Plot individual assets
        if show_assets:
            asset_vols = np.sqrt(np.diag(self.cov_matrix)) * 100
            asset_rets = self.expected_returns * 100

            ax.scatter(
                asset_vols,
                asset_rets,
                c='gray',
                marker='o',
                s=100,
                alpha=0.7,
                label='Individual Assets'
            )

            # Label assets
            for i, name in enumerate(self.asset_names):
                ax.annotate(
                    name,
                    (asset_vols[i], asset_rets[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )

        # Plot special portfolios
        if show_special_portfolios:
            min_var = frontier[frontier['type'] == 'min_variance']
            if len(min_var) > 0:
                ax.scatter(
                    min_var['volatility'].iloc[0] * 100,
                    min_var['return'].iloc[0] * 100,
                    c='green',
                    marker='*',
                    s=200,
                    zorder=5,
                    label='Min Variance'
                )

            max_sharpe = frontier[frontier['type'] == 'max_sharpe']
            if len(max_sharpe) > 0:
                ax.scatter(
                    max_sharpe['volatility'].iloc[0] * 100,
                    max_sharpe['return'].iloc[0] * 100,
                    c='red',
                    marker='*',
                    s=200,
                    zorder=5,
                    label='Max Sharpe'
                )

        # Format plot
        ax.set_xlabel('Volatility (%)', fontsize=12)
        ax.set_ylabel('Expected Return (%)', fontsize=12)
        ax.set_title('Efficient Frontier', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        return ax

    def compare_strategies(
        self,
        constraints: Optional[PortfolioConstraints] = None
    ) -> pd.DataFrame:
        """
        Compare different optimization strategies.

        Calculates and compares portfolios using different optimization
        approaches to help select the best strategy.

        Args:
            constraints: PortfolioConstraints object (uses defaults if None)

        Returns:
            DataFrame comparing different optimization strategies
        """
        strategies = {}

        # Equal weight
        equal_weights = np.ones(self.n_assets) / self.n_assets
        strategies['Equal Weight'] = self.calculate_portfolio_metrics(equal_weights)

        # Min variance
        strategies['Min Variance'] = self.optimize_min_variance(constraints)

        # Max Sharpe
        strategies['Max Sharpe'] = self.optimize_max_sharpe(constraints)

        # Risk Parity
        rp_weights = self.risk_parity_weights(constraints)
        strategies['Risk Parity'] = self.calculate_portfolio_metrics(rp_weights)

        # HRP
        hrp_weights = self.hierarchical_risk_parity()
        strategies['HRP'] = self.calculate_portfolio_metrics(hrp_weights)

        # Max Diversification
        strategies['Max Diversification'] = self._optimize_max_diversification(constraints)

        # Create comparison DataFrame
        comparison = []
        for name, metrics in strategies.items():
            comparison.append({
                'Strategy': name,
                'Return': f"{metrics.expected_return:.2%}",
                'Volatility': f"{metrics.volatility:.2%}",
                'Sharpe': f"{metrics.sharpe_ratio:.3f}",
                'Div Ratio': f"{metrics.diversification_ratio:.3f}",
                'Effective N': f"{metrics.effective_n:.2f}",
            })

        return pd.DataFrame(comparison)


# =============================================================================
# CVaR (CONDITIONAL VALUE AT RISK) OPTIMIZATION
# =============================================================================

@dataclass
class CVaRResult:
    """
    Results from CVaR optimization.

    Attributes:
        weights: Optimal portfolio weights
        expected_return: Portfolio expected return
        var: Value at Risk at specified confidence level
        cvar: Conditional Value at Risk (Expected Shortfall)
        cvar_sharpe: Return per unit of CVaR risk
        optimization_success: Whether optimization converged
    """
    weights: np.ndarray
    expected_return: float
    var: float
    cvar: float
    cvar_sharpe: float
    optimization_success: bool


class CVaROptimizer:
    """
    CVaR (Conditional Value at Risk) Portfolio Optimization.

    CVaR, also known as Expected Shortfall (ES), measures the expected loss
    in the worst α% of cases. Unlike VaR, CVaR is a coherent risk measure
    satisfying subadditivity, making it theoretically superior for optimization.

    This implementation uses the Rockafellar-Uryasev (2000) linear programming
    formulation, which converts CVaR optimization into a convex problem.

    ★ Insight ─────────────────────────────────────
    Why CVaR over VaR?
    - VaR only tells you the threshold; CVaR tells you expected loss beyond it
    - CVaR is convex and subadditive (can be optimized reliably)
    - Better captures tail risk in non-normal return distributions
    - Preferred by Basel III/IV for internal models
    ─────────────────────────────────────────────────

    Attributes:
        returns: Historical return scenarios (DataFrame or array)
        confidence_level: CVaR confidence level (e.g., 0.95 for 95% CVaR)
        risk_free_rate: Annual risk-free rate for ratio calculations
        n_assets: Number of assets
        n_scenarios: Number of historical scenarios

    Example:
        >>> optimizer = CVaROptimizer(returns, confidence_level=0.95)
        >>> result = optimizer.min_cvar(target_return=0.10)
        >>> print(f"CVaR: {result.cvar:.2%}")
        >>> print(f"Weights: {result.weights}")

    References:
        - Rockafellar, R.T. and Uryasev, S. (2000). "Optimization of Conditional
          Value-at-Risk". Journal of Risk.
        - Pflug, G. (2000). "Some Remarks on the Value-at-Risk and the
          Conditional Value-at-Risk". Probabilistic Constrained Optimization.
    """

    def __init__(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.02,
        annualization_factor: int = 252
    ):
        """
        Initialize CVaR optimizer.

        Args:
            returns: Historical returns (rows=scenarios, cols=assets)
            confidence_level: CVaR confidence level (0.90, 0.95, 0.99)
            risk_free_rate: Annual risk-free rate
            annualization_factor: Factor to annualize returns
        """
        if isinstance(returns, pd.DataFrame):
            self.returns = returns.values
            self.asset_names = list(returns.columns)
        else:
            self.returns = np.array(returns)
            self.asset_names = [f"Asset_{i}" for i in range(returns.shape[1])]

        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        self.annualization_factor = annualization_factor
        self.n_scenarios, self.n_assets = self.returns.shape

        # Pre-compute expected returns
        self.expected_returns = np.mean(self.returns, axis=0) * annualization_factor

    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate annualized portfolio expected return."""
        return np.dot(weights, self.expected_returns)

    def portfolio_var(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio VaR using historical simulation.

        Args:
            weights: Portfolio weights

        Returns:
            VaR (positive number representing loss)
        """
        portfolio_returns = self.returns @ weights
        var_percentile = (1 - self.confidence_level) * 100
        var = -np.percentile(portfolio_returns, var_percentile)
        return var * np.sqrt(self.annualization_factor)

    def portfolio_cvar(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio CVaR (Expected Shortfall) using historical simulation.

        CVaR = E[Loss | Loss > VaR]

        Args:
            weights: Portfolio weights

        Returns:
            CVaR (positive number representing expected loss in tail)
        """
        portfolio_returns = self.returns @ weights
        var_threshold = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        tail_returns = portfolio_returns[portfolio_returns <= var_threshold]

        if len(tail_returns) == 0:
            return -var_threshold * np.sqrt(self.annualization_factor)

        cvar = -np.mean(tail_returns)
        return cvar * np.sqrt(self.annualization_factor)

    def min_cvar(
        self,
        target_return: Optional[float] = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        long_only: bool = True,
        group_constraints: Optional[Dict[str, Tuple[List[int], float, float]]] = None
    ) -> CVaRResult:
        """
        Minimize CVaR subject to constraints.

        Uses the Rockafellar-Uryasev formulation:

        min  ζ + (1/(1-α)) * (1/S) * Σ u_s
        s.t. u_s ≥ -r_s'w - ζ   ∀s
             u_s ≥ 0            ∀s
             w'1 = 1
             w ≥ 0 (if long_only)
             w'μ ≥ target_return (if specified)

        where ζ is VaR and u_s are auxiliary variables for tail losses.

        Args:
            target_return: Optional minimum return constraint (annualized)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            long_only: If True, no short positions
            group_constraints: Dict of group constraints:
                             {name: (asset_indices, min_exposure, max_exposure)}

        Returns:
            CVaRResult with optimal weights and risk metrics

        Example:
            >>> result = optimizer.min_cvar(target_return=0.10, max_weight=0.25)
            >>> print(f"Min CVaR at 10% return: {result.cvar:.2%}")
        """
        from scipy.optimize import linprog, minimize

        # Number of variables: n_assets weights + 1 zeta + n_scenarios auxiliary
        n_vars = self.n_assets + 1 + self.n_scenarios

        # Objective: minimize ζ + (1/(1-α)) * (1/S) * Σ u_s
        c = np.zeros(n_vars)
        c[self.n_assets] = 1  # Coefficient for ζ
        tail_weight = 1 / ((1 - self.confidence_level) * self.n_scenarios)
        c[self.n_assets + 1:] = tail_weight  # Coefficients for u_s

        # Build inequality constraints: u_s ≥ -r_s'w - ζ → r_s'w + ζ + u_s ≥ 0
        # In standard form: A_ub @ x ≤ b_ub → -r_s'w - ζ - u_s ≤ 0
        A_ub = np.zeros((self.n_scenarios, n_vars))
        for s in range(self.n_scenarios):
            A_ub[s, :self.n_assets] = -self.returns[s]  # -r_s
            A_ub[s, self.n_assets] = -1  # -ζ
            A_ub[s, self.n_assets + 1 + s] = -1  # -u_s
        b_ub = np.zeros(self.n_scenarios)

        # Equality constraint: sum of weights = 1
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :self.n_assets] = 1
        b_eq = np.array([1.0])

        # Add target return constraint if specified
        if target_return is not None:
            # w'μ ≥ target → -w'μ ≤ -target (daily)
            daily_target = target_return / self.annualization_factor
            return_row = np.zeros((1, n_vars))
            return_row[0, :self.n_assets] = -np.mean(self.returns, axis=0)
            A_ub = np.vstack([A_ub, return_row])
            b_ub = np.append(b_ub, -daily_target)

        # Add group constraints
        if group_constraints is not None:
            for name, (indices, min_exp, max_exp) in group_constraints.items():
                # Min exposure: sum(w_i for i in group) >= min_exp → -sum ≤ -min_exp
                min_row = np.zeros((1, n_vars))
                min_row[0, indices] = -1
                A_ub = np.vstack([A_ub, min_row])
                b_ub = np.append(b_ub, -min_exp)

                # Max exposure: sum(w_i for i in group) <= max_exp
                max_row = np.zeros((1, n_vars))
                max_row[0, indices] = 1
                A_ub = np.vstack([A_ub, max_row])
                b_ub = np.append(b_ub, max_exp)

        # Bounds
        lb = np.zeros(n_vars)
        ub = np.full(n_vars, np.inf)

        # Weight bounds
        if long_only:
            lb[:self.n_assets] = max(0, min_weight)
        else:
            lb[:self.n_assets] = min_weight
        ub[:self.n_assets] = max_weight

        # ζ can be negative (VaR threshold)
        lb[self.n_assets] = -np.inf

        # u_s ≥ 0 already set

        bounds = list(zip(lb, ub))

        # Solve LP
        try:
            result = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method='highs'
            )

            if result.success:
                weights = result.x[:self.n_assets]
                zeta = result.x[self.n_assets]

                # Calculate actual CVaR from solution
                cvar = self.portfolio_cvar(weights)
                var = self.portfolio_var(weights)
                expected_ret = self.portfolio_return(weights)

                # CVaR Sharpe ratio
                cvar_sharpe = (expected_ret - self.risk_free_rate) / cvar if cvar > 0 else 0

                return CVaRResult(
                    weights=weights,
                    expected_return=expected_ret,
                    var=var,
                    cvar=cvar,
                    cvar_sharpe=cvar_sharpe,
                    optimization_success=True
                )
        except Exception as e:
            warnings.warn(f"LP optimization failed: {e}, falling back to SLSQP")

        # Fallback to direct optimization
        return self._min_cvar_slsqp(target_return, max_weight, min_weight, long_only)

    def _min_cvar_slsqp(
        self,
        target_return: Optional[float],
        max_weight: float,
        min_weight: float,
        long_only: bool
    ) -> CVaRResult:
        """Fallback SLSQP optimization for CVaR."""
        def objective(w):
            return self.portfolio_cvar(w)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        if target_return is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, tr=target_return: self.portfolio_return(w) - tr
            })

        lb = max(0, min_weight) if long_only else min_weight
        bounds = [(lb, max_weight) for _ in range(self.n_assets)]

        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )

        weights = result.x
        cvar = self.portfolio_cvar(weights)
        var = self.portfolio_var(weights)
        expected_ret = self.portfolio_return(weights)
        cvar_sharpe = (expected_ret - self.risk_free_rate) / cvar if cvar > 0 else 0

        return CVaRResult(
            weights=weights,
            expected_return=expected_ret,
            var=var,
            cvar=cvar,
            cvar_sharpe=cvar_sharpe,
            optimization_success=result.success
        )

    def max_return_for_cvar(
        self,
        target_cvar: float,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        long_only: bool = True
    ) -> CVaRResult:
        """
        Maximize return subject to CVaR constraint.

        Find the portfolio with highest expected return while keeping
        CVaR below a specified threshold.

        Args:
            target_cvar: Maximum allowed CVaR (annualized)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            long_only: If True, no short positions

        Returns:
            CVaRResult with optimal weights

        Example:
            >>> result = optimizer.max_return_for_cvar(target_cvar=0.15)
            >>> print(f"Max return at 15% CVaR: {result.expected_return:.2%}")
        """
        def objective(w):
            return -self.portfolio_return(w)  # Negative for maximization

        def cvar_constraint(w):
            return target_cvar - self.portfolio_cvar(w)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': cvar_constraint}
        ]

        lb = max(0, min_weight) if long_only else min_weight
        bounds = [(lb, max_weight) for _ in range(self.n_assets)]

        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )

        weights = result.x
        cvar = self.portfolio_cvar(weights)
        var = self.portfolio_var(weights)
        expected_ret = self.portfolio_return(weights)
        cvar_sharpe = (expected_ret - self.risk_free_rate) / cvar if cvar > 0 else 0

        return CVaRResult(
            weights=weights,
            expected_return=expected_ret,
            var=var,
            cvar=cvar,
            cvar_sharpe=cvar_sharpe,
            optimization_success=result.success
        )

    def cvar_efficient_frontier(
        self,
        n_points: int = 20,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        long_only: bool = True
    ) -> pd.DataFrame:
        """
        Generate CVaR-efficient frontier.

        Creates a frontier showing the optimal return-CVaR tradeoff,
        analogous to the mean-variance efficient frontier.

        Args:
            n_points: Number of points on the frontier
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            long_only: If True, no short positions

        Returns:
            DataFrame with columns: return, cvar, var, cvar_sharpe, weights

        Example:
            >>> frontier = optimizer.cvar_efficient_frontier(n_points=50)
            >>> frontier.plot(x='cvar', y='return')
        """
        # Find min CVaR portfolio (no return constraint)
        min_cvar_result = self.min_cvar(
            target_return=None,
            max_weight=max_weight,
            min_weight=min_weight,
            long_only=long_only
        )
        min_return = min_cvar_result.expected_return

        # Find max return portfolio
        max_return = np.max(self.expected_returns)

        # Generate target returns
        target_returns = np.linspace(min_return, max_return * 0.95, n_points)

        frontier_data = []
        for target in target_returns:
            try:
                result = self.min_cvar(
                    target_return=target,
                    max_weight=max_weight,
                    min_weight=min_weight,
                    long_only=long_only
                )

                if result.optimization_success:
                    frontier_data.append({
                        'return': result.expected_return,
                        'cvar': result.cvar,
                        'var': result.var,
                        'cvar_sharpe': result.cvar_sharpe,
                        'weights': result.weights.tolist()
                    })
            except Exception:
                continue

        return pd.DataFrame(frontier_data).sort_values('cvar').reset_index(drop=True)

    def cvar_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate CVaR-adjusted Sharpe ratio.

        Analogous to Sharpe ratio but using CVaR instead of volatility:
        CVaR Sharpe = (E[R] - Rf) / CVaR

        Args:
            weights: Portfolio weights

        Returns:
            CVaR Sharpe ratio
        """
        expected_ret = self.portfolio_return(weights)
        cvar = self.portfolio_cvar(weights)

        if cvar <= 0:
            return 0

        return (expected_ret - self.risk_free_rate) / cvar

    def max_cvar_sharpe(
        self,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        long_only: bool = True
    ) -> CVaRResult:
        """
        Find portfolio maximizing CVaR Sharpe ratio.

        Analogous to finding the tangency portfolio in mean-variance,
        but using CVaR as the risk measure.

        Args:
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            long_only: If True, no short positions

        Returns:
            CVaRResult with maximum CVaR Sharpe portfolio
        """
        def neg_cvar_sharpe(w):
            expected_ret = self.portfolio_return(w)
            cvar = self.portfolio_cvar(w)
            if cvar <= 0:
                return 0
            return -(expected_ret - self.risk_free_rate) / cvar

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        lb = max(0, min_weight) if long_only else min_weight
        bounds = [(lb, max_weight) for _ in range(self.n_assets)]

        x0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            neg_cvar_sharpe, x0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'ftol': 1e-10, 'maxiter': 1000}
        )

        weights = result.x
        cvar = self.portfolio_cvar(weights)
        var = self.portfolio_var(weights)
        expected_ret = self.portfolio_return(weights)
        cvar_sharpe = (expected_ret - self.risk_free_rate) / cvar if cvar > 0 else 0

        return CVaRResult(
            weights=weights,
            expected_return=expected_ret,
            var=var,
            cvar=cvar,
            cvar_sharpe=cvar_sharpe,
            optimization_success=result.success
        )

    def scenario_analysis(
        self,
        weights: np.ndarray,
        worst_n: int = 10
    ) -> Dict:
        """
        Analyze portfolio performance in worst scenarios.

        Provides detailed analysis of portfolio returns in the worst
        historical scenarios to understand tail risk characteristics.

        Args:
            weights: Portfolio weights to analyze
            worst_n: Number of worst scenarios to report

        Returns:
            Dictionary with:
                - worst_scenarios: Details of worst N scenarios
                - tail_statistics: Statistics of tail returns
                - drawdown_analysis: Drawdown characteristics
        """
        portfolio_returns = self.returns @ weights
        sorted_returns = np.sort(portfolio_returns)

        # VaR threshold
        var_idx = int(np.floor((1 - self.confidence_level) * self.n_scenarios))
        var_threshold = sorted_returns[var_idx]

        # Worst scenarios
        worst_indices = np.argsort(portfolio_returns)[:worst_n]
        worst_returns = portfolio_returns[worst_indices]

        # Tail statistics
        tail_returns = sorted_returns[:var_idx + 1]

        # Contribution to CVaR by asset
        tail_mask = portfolio_returns <= var_threshold
        if np.sum(tail_mask) > 0:
            tail_scenarios = self.returns[tail_mask]
            weighted_tail = tail_scenarios * weights
            cvar_contribution = np.mean(weighted_tail, axis=0)
            cvar_contribution_pct = cvar_contribution / np.sum(np.abs(cvar_contribution))
        else:
            cvar_contribution_pct = weights

        return {
            'worst_scenarios': {
                'indices': worst_indices.tolist(),
                'returns': (worst_returns * np.sqrt(self.annualization_factor)).tolist(),
                'annualized_impact': float(np.mean(worst_returns) * self.annualization_factor)
            },
            'tail_statistics': {
                'var': float(-var_threshold * np.sqrt(self.annualization_factor)),
                'cvar': float(-np.mean(tail_returns) * np.sqrt(self.annualization_factor)),
                'tail_skewness': float(pd.Series(tail_returns).skew()),
                'tail_kurtosis': float(pd.Series(tail_returns).kurtosis()),
                'n_tail_scenarios': len(tail_returns)
            },
            'cvar_contribution': dict(zip(self.asset_names, cvar_contribution_pct))
        }

    def compare_with_mean_variance(
        self,
        target_return: float,
        markowitz_optimizer: 'MarkowitzOptimizer'
    ) -> pd.DataFrame:
        """
        Compare CVaR optimal portfolio with mean-variance optimal.

        Provides side-by-side comparison to understand the difference
        between volatility-based and tail-risk-based optimization.

        Args:
            target_return: Target return for both optimizations
            markowitz_optimizer: MarkowitzOptimizer instance with same assets

        Returns:
            DataFrame comparing the two approaches
        """
        # CVaR optimal
        cvar_result = self.min_cvar(target_return=target_return)

        # Mean-variance optimal
        mv_metrics = markowitz_optimizer.optimize_target_return(target_return)

        # Calculate both risk measures for both portfolios
        cvar_mv = self.portfolio_cvar(mv_metrics.weights)
        vol_cvar = markowitz_optimizer._portfolio_volatility(cvar_result.weights)

        comparison = pd.DataFrame({
            'Metric': [
                'Expected Return',
                'Volatility',
                'VaR (95%)',
                'CVaR (95%)',
                'Sharpe Ratio',
                'CVaR Sharpe'
            ],
            'CVaR Optimal': [
                f"{cvar_result.expected_return:.2%}",
                f"{vol_cvar:.2%}",
                f"{cvar_result.var:.2%}",
                f"{cvar_result.cvar:.2%}",
                f"{(cvar_result.expected_return - self.risk_free_rate) / vol_cvar:.3f}",
                f"{cvar_result.cvar_sharpe:.3f}"
            ],
            'Mean-Variance Optimal': [
                f"{mv_metrics.expected_return:.2%}",
                f"{mv_metrics.volatility:.2%}",
                f"{self.portfolio_var(mv_metrics.weights):.2%}",
                f"{cvar_mv:.2%}",
                f"{mv_metrics.sharpe_ratio:.3f}",
                f"{(mv_metrics.expected_return - self.risk_free_rate) / cvar_mv:.3f}"
            ]
        })

        return comparison


def create_sample_returns(
    n_assets: int = 10,
    n_days: int = 252,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create sample returns data for testing.

    Generates synthetic return data with realistic correlation structure
    for testing and demonstration purposes.

    Args:
        n_assets: Number of assets to generate
        n_days: Number of trading days
        seed: Random seed for reproducibility

    Returns:
        DataFrame of simulated daily returns
    """
    np.random.seed(seed)

    # Generate asset names
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]

    # Generate correlation matrix with some structure
    base_corr = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            # Higher correlation within "sectors" (groups of 3)
            if i // 3 == j // 3:
                corr = np.random.uniform(0.3, 0.7)
            else:
                corr = np.random.uniform(-0.2, 0.3)
            base_corr[i, j] = corr
            base_corr[j, i] = corr

    # Generate volatilities (annual)
    vols = np.random.uniform(0.15, 0.40, n_assets)

    # Create covariance matrix
    D = np.diag(vols)
    cov = D @ base_corr @ D

    # Daily covariance
    daily_cov = cov / 252

    # Cholesky decomposition for correlated returns
    L = np.linalg.cholesky(daily_cov)

    # Generate returns
    random_returns = np.random.randn(n_days, n_assets)
    returns = random_returns @ L.T

    # Add small positive drift (expected return)
    drift = np.random.uniform(0.05, 0.15, n_assets) / 252
    returns += drift

    return pd.DataFrame(returns, columns=asset_names)


# Convenience function for quick optimization
def optimize_portfolio(
    returns: pd.DataFrame,
    objective: str = 'max_sharpe',
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Quick portfolio optimization function.

    Convenience wrapper for common optimization scenarios.

    Args:
        returns: DataFrame of asset returns
        objective: One of 'max_sharpe', 'min_variance', 'risk_parity', 'hrp'
        long_only: If True, no short positions allowed
        max_weight: Maximum weight per asset
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of optimal weights

    Example:
        >>> weights = optimize_portfolio(returns, objective='max_sharpe', max_weight=0.3)
    """
    optimizer = MarkowitzOptimizer(returns, risk_free_rate=risk_free_rate)

    constraints = PortfolioConstraints(
        long_only=long_only,
        max_weight=max_weight
    )

    objective_map = {
        'max_sharpe': OptimizationObjective.MAX_SHARPE,
        'min_variance': OptimizationObjective.MIN_VARIANCE,
        'risk_parity': OptimizationObjective.RISK_PARITY,
    }

    if objective == 'hrp':
        weights = optimizer.hierarchical_risk_parity()
        return dict(zip(optimizer.asset_names, weights))

    if objective not in objective_map:
        raise ValueError(f"Unknown objective: {objective}. Choose from {list(objective_map.keys()) + ['hrp']}")

    return optimizer.get_optimal_weights(
        objective=objective_map[objective],
        constraints=constraints
    )
