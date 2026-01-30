"""
Fama-French Factor Model Strategy for Quantsploit

This module implements factor-based investing strategies using the
Fama-French factor framework. Supports factor construction, exposure
analysis, and factor-tilted portfolio construction.

Key Features:
- Multi-factor scoring (Market, Size, Value, Momentum, Profitability, Investment)
- Factor exposure calculation and monitoring
- Factor-tilted portfolio construction
- Factor momentum strategies
- Integration with backtesting framework

References:
    - Fama, E. & French, K. (1993). "Common Risk Factors in Stock Returns"
    - Fama, E. & French, K. (2015). "A Five-Factor Asset Pricing Model"
    - Carhart, M. (1997). "On Persistence in Mutual Fund Performance"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """Fama-French and extended factors"""
    MKT = "market"          # Market excess return
    SMB = "size"            # Small Minus Big
    HML = "value"           # High Minus Low (Book-to-Market)
    MOM = "momentum"        # Winners Minus Losers (Carhart)
    RMW = "profitability"   # Robust Minus Weak
    CMA = "investment"      # Conservative Minus Aggressive


@dataclass
class FactorExposure:
    """
    Factor exposure analysis results.

    Attributes:
        betas: Factor betas/loadings
        t_stats: T-statistics for each beta
        p_values: P-values for each beta
        r_squared: Regression R-squared
        alpha: Jensen's alpha
        alpha_t_stat: T-statistic for alpha
        residual_vol: Residual volatility
    """
    betas: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    alpha: float
    alpha_t_stat: float
    residual_vol: float

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "Factor Exposure Analysis",
            "=" * 50,
            f"R-squared: {self.r_squared:.4f}",
            f"Alpha: {self.alpha:.4f} (t={self.alpha_t_stat:.2f})",
            f"Residual Vol: {self.residual_vol:.4f}",
            "",
            f"{'Factor':<15} {'Beta':>10} {'t-stat':>10} {'p-value':>10}"
        ]

        for factor in self.betas:
            lines.append(
                f"{factor:<15} {self.betas[factor]:>10.4f} "
                f"{self.t_stats[factor]:>10.2f} {self.p_values[factor]:>10.4f}"
            )

        return "\n".join(lines)


@dataclass
class FactorScore:
    """
    Factor scores for an asset.

    Attributes:
        symbol: Asset symbol
        scores: Dictionary of factor scores
        composite_score: Combined factor score
        rank: Percentile rank
    """
    symbol: str
    scores: Dict[str, float]
    composite_score: float
    rank: float


class FamaFrenchStrategy:
    """
    Fama-French Factor Model Strategy.

    Implements factor-based portfolio construction and analysis using
    the Fama-French multi-factor framework. Can calculate factor exposures,
    construct factor-tilted portfolios, and run factor timing strategies.

    ★ Insight ─────────────────────────────────────
    The Five Factors (Fama-French 2015):
    - MKT: Market risk premium (equity vs risk-free)
    - SMB: Size premium (small cap outperformance)
    - HML: Value premium (cheap stocks beat expensive)
    - RMW: Profitability (profitable firms outperform)
    - CMA: Investment (conservative investors beat aggressive)
    Plus Momentum (MOM) from Carhart (1997)
    ─────────────────────────────────────────────────

    Example:
        >>> strategy = FamaFrenchStrategy(stock_returns, factor_returns)
        >>> exposure = strategy.calculate_factor_exposure('AAPL')
        >>> print(exposure.summary())

    Attributes:
        stock_returns: DataFrame of stock returns
        factor_returns: DataFrame of factor returns
        risk_free_rate: Risk-free rate series
    """

    def __init__(
        self,
        stock_returns: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None,
        risk_free_rate: Optional[pd.Series] = None,
        annualization_factor: int = 252
    ):
        """
        Initialize Fama-French Strategy.

        Args:
            stock_returns: DataFrame of individual stock returns
            factor_returns: DataFrame of factor returns (columns: MKT, SMB, HML, etc.)
                          If None, factors will be constructed from stock_returns
            risk_free_rate: Series of risk-free rates
            annualization_factor: Factor for annualization
        """
        self.stock_returns = stock_returns
        self.factor_returns = factor_returns
        self.risk_free_rate = risk_free_rate if risk_free_rate is not None else pd.Series(0.0, index=stock_returns.index)
        self.annualization_factor = annualization_factor

        # Align indices
        if factor_returns is not None:
            common_idx = stock_returns.index.intersection(factor_returns.index)
            self.stock_returns = stock_returns.loc[common_idx]
            self.factor_returns = factor_returns.loc[common_idx]
            self.risk_free_rate = self.risk_free_rate.loc[common_idx]

    def calculate_factor_exposure(
        self,
        asset: str,
        factors: Optional[List[str]] = None
    ) -> FactorExposure:
        """
        Calculate factor exposures for a single asset.

        Uses OLS regression to estimate factor betas:
        R_i - Rf = alpha + beta_1*MKT + beta_2*SMB + ... + epsilon

        Args:
            asset: Asset name/column
            factors: List of factor names to include (default: all available)

        Returns:
            FactorExposure with betas and statistics
        """
        if self.factor_returns is None:
            raise ValueError("Factor returns required for exposure calculation")

        # Get asset returns
        y = self.stock_returns[asset] - self.risk_free_rate.values

        # Select factors
        if factors is None:
            factors = list(self.factor_returns.columns)

        X = self.factor_returns[factors].values
        X = np.column_stack([np.ones(len(X)), X])  # Add constant for alpha

        # OLS regression
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X, y.values, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            beta = np.linalg.pinv(X) @ y.values
            residuals = y.values - X @ beta

        # Calculate statistics
        n = len(y)
        k = len(factors) + 1  # factors + alpha

        # Residuals
        y_hat = X @ beta
        resid = y.values - y_hat
        resid_var = np.var(resid, ddof=k)

        # Standard errors
        try:
            cov_matrix = resid_var * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov_matrix))
        except np.linalg.LinAlgError:
            se = np.ones(k) * np.inf

        # T-statistics
        t_stats = beta / se

        # P-values (two-tailed)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

        # R-squared
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y.values - np.mean(y.values)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Create result
        betas_dict = {f: beta[i + 1] for i, f in enumerate(factors)}
        t_stats_dict = {f: t_stats[i + 1] for i, f in enumerate(factors)}
        p_values_dict = {f: p_values[i + 1] for i, f in enumerate(factors)}

        return FactorExposure(
            betas=betas_dict,
            t_stats=t_stats_dict,
            p_values=p_values_dict,
            r_squared=r_squared,
            alpha=beta[0] * self.annualization_factor,  # Annualize alpha
            alpha_t_stat=t_stats[0],
            residual_vol=np.std(resid) * np.sqrt(self.annualization_factor)
        )

    def calculate_all_exposures(
        self,
        factors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate factor exposures for all stocks.

        Args:
            factors: List of factors to include

        Returns:
            DataFrame with exposures for each stock
        """
        exposures = []

        for asset in self.stock_returns.columns:
            try:
                exp = self.calculate_factor_exposure(asset, factors)
                row = {'asset': asset, 'alpha': exp.alpha, 'r_squared': exp.r_squared}
                row.update(exp.betas)
                exposures.append(row)
            except Exception as e:
                logger.warning(f"Could not calculate exposure for {asset}: {e}")

        return pd.DataFrame(exposures).set_index('asset')

    def construct_factor_portfolios(
        self,
        characteristics: pd.DataFrame,
        n_quantiles: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Construct factor portfolios from stock characteristics.

        Creates long-short portfolios based on characteristic sorts.

        Args:
            characteristics: DataFrame with columns for each characteristic
                           (e.g., 'market_cap', 'book_to_market', 'momentum')
            n_quantiles: Number of quantiles for sorting

        Returns:
            Dictionary of factor return DataFrames
        """
        factor_portfolios = {}

        # Size factor (SMB)
        if 'market_cap' in characteristics.columns:
            factor_portfolios['SMB'] = self._construct_long_short(
                characteristic=characteristics['market_cap'],
                long_quantile=1,  # Small
                short_quantile=n_quantiles  # Big
            )

        # Value factor (HML)
        if 'book_to_market' in characteristics.columns:
            factor_portfolios['HML'] = self._construct_long_short(
                characteristic=characteristics['book_to_market'],
                long_quantile=n_quantiles,  # High B/M (Value)
                short_quantile=1  # Low B/M (Growth)
            )

        # Momentum factor (MOM)
        if 'momentum' in characteristics.columns:
            factor_portfolios['MOM'] = self._construct_long_short(
                characteristic=characteristics['momentum'],
                long_quantile=n_quantiles,  # Winners
                short_quantile=1  # Losers
            )

        # Profitability factor (RMW)
        if 'profitability' in characteristics.columns:
            factor_portfolios['RMW'] = self._construct_long_short(
                characteristic=characteristics['profitability'],
                long_quantile=n_quantiles,  # Robust
                short_quantile=1  # Weak
            )

        # Investment factor (CMA)
        if 'investment' in characteristics.columns:
            factor_portfolios['CMA'] = self._construct_long_short(
                characteristic=characteristics['investment'],
                long_quantile=1,  # Conservative
                short_quantile=n_quantiles  # Aggressive
            )

        return factor_portfolios

    def _construct_long_short(
        self,
        characteristic: pd.Series,
        long_quantile: int,
        short_quantile: int,
        n_quantiles: int = 5
    ) -> pd.Series:
        """
        Construct long-short portfolio returns from characteristic.

        Args:
            characteristic: Series of characteristic values
            long_quantile: Quantile to go long
            short_quantile: Quantile to go short
            n_quantiles: Total number of quantiles

        Returns:
            Series of long-short returns
        """
        # Rank into quantiles
        quantiles = pd.qcut(characteristic, n_quantiles, labels=False, duplicates='drop')

        # Get stocks in each leg
        long_stocks = characteristic.index[quantiles == long_quantile - 1]
        short_stocks = characteristic.index[quantiles == short_quantile - 1]

        # Equal-weight returns
        long_returns = self.stock_returns[long_stocks].mean(axis=1)
        short_returns = self.stock_returns[short_stocks].mean(axis=1)

        return long_returns - short_returns

    def calculate_factor_scores(
        self,
        characteristics: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> List[FactorScore]:
        """
        Calculate factor scores for all assets.

        Args:
            characteristics: DataFrame of factor characteristics
            weights: Optional weights for combining factors

        Returns:
            List of FactorScore objects sorted by composite score
        """
        if weights is None:
            # Equal weights by default
            weights = {col: 1.0 / len(characteristics.columns)
                      for col in characteristics.columns}

        # Standardize characteristics
        standardized = (characteristics - characteristics.mean()) / characteristics.std()

        scores = []
        for asset in standardized.index:
            asset_scores = {}
            composite = 0

            for factor in standardized.columns:
                score = standardized.loc[asset, factor]
                if not np.isnan(score):
                    asset_scores[factor] = score
                    composite += weights.get(factor, 0) * score

            scores.append(FactorScore(
                symbol=asset,
                scores=asset_scores,
                composite_score=composite,
                rank=0  # Will be set after sorting
            ))

        # Sort and assign ranks
        scores.sort(key=lambda x: x.composite_score, reverse=True)
        for i, score in enumerate(scores):
            score.rank = (i + 1) / len(scores)

        return scores

    def construct_factor_tilted_portfolio(
        self,
        target_exposures: Dict[str, float],
        constraints: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Construct portfolio with target factor exposures.

        Uses optimization to find weights that achieve target factor loadings
        while satisfying constraints.

        Args:
            target_exposures: Target factor exposures {factor: target_beta}
            constraints: Optional constraints dict with:
                       - max_weight: Maximum weight per stock
                       - min_weight: Minimum weight per stock
                       - long_only: Boolean for long-only constraint

        Returns:
            Array of portfolio weights
        """
        from scipy.optimize import minimize

        if self.factor_returns is None:
            raise ValueError("Factor returns required")

        n_assets = len(self.stock_returns.columns)
        factors = list(target_exposures.keys())

        # Calculate stock factor betas
        stock_betas = np.zeros((n_assets, len(factors)))
        for i, asset in enumerate(self.stock_returns.columns):
            try:
                exposure = self.calculate_factor_exposure(asset, factors)
                for j, f in enumerate(factors):
                    stock_betas[i, j] = exposure.betas.get(f, 0)
            except Exception:
                pass

        targets = np.array([target_exposures[f] for f in factors])

        # Objective: minimize squared deviation from target exposures
        def objective(weights):
            portfolio_betas = stock_betas.T @ weights
            return np.sum((portfolio_betas - targets) ** 2)

        # Constraints
        if constraints is None:
            constraints = {}

        max_weight = constraints.get('max_weight', 1.0)
        min_weight = constraints.get('min_weight', 0.0)
        long_only = constraints.get('long_only', True)

        if long_only:
            bounds = [(max(0, min_weight), max_weight) for _ in range(n_assets)]
        else:
            bounds = [(min_weight, max_weight) for _ in range(n_assets)]

        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # Initial guess: equal weight
        x0 = np.ones(n_assets) / n_assets

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

        return result.x

    def run_factor_momentum_strategy(
        self,
        lookback: int = 12,
        holding_period: int = 1,
        n_factors: int = 3
    ) -> Dict:
        """
        Run factor momentum strategy.

        Selects factors with strongest recent performance and constructs
        a portfolio tilted towards those factors.

        Args:
            lookback: Months for momentum calculation
            holding_period: Months to hold position
            n_factors: Number of top factors to include

        Returns:
            Dictionary with strategy results
        """
        if self.factor_returns is None:
            raise ValueError("Factor returns required")

        factors = list(self.factor_returns.columns)
        dates = self.factor_returns.index

        # Resample to monthly for factor momentum
        monthly_factor = self.factor_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        portfolio_returns = []
        selected_factors_history = []

        for i in range(lookback, len(monthly_factor)):
            # Calculate factor momentum
            lookback_returns = monthly_factor.iloc[i - lookback:i]
            factor_mom = lookback_returns.sum()

            # Select top N factors
            top_factors = factor_mom.nlargest(n_factors).index.tolist()
            selected_factors_history.append(top_factors)

            # Equal-weight top factors
            if i < len(monthly_factor):
                port_ret = monthly_factor.iloc[i][top_factors].mean()
                portfolio_returns.append(port_ret)

        portfolio_returns = np.array(portfolio_returns)

        # Calculate metrics
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        annualized_return = (1 + cumulative_return) ** (12 / len(portfolio_returns)) - 1
        annualized_vol = np.std(portfolio_returns) * np.sqrt(12)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

        return {
            'returns': portfolio_returns,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe,
            'selected_factors': selected_factors_history
        }

    def run_backtest(
        self,
        method: str = 'factor_score',
        rebalance_frequency: int = 21,
        n_stocks: int = 50,
        initial_capital: float = 100000.0,
        **kwargs
    ) -> Dict:
        """
        Backtest factor strategy.

        Args:
            method: Strategy method ('factor_score', 'factor_tilt')
            rebalance_frequency: Days between rebalances
            n_stocks: Number of stocks in portfolio
            initial_capital: Starting capital
            **kwargs: Additional arguments

        Returns:
            Dictionary with backtest results
        """
        # For now, implement simple factor-score based strategy
        # using historical returns as momentum proxy

        dates = self.stock_returns.index
        n_days = len(dates)

        portfolio_returns = []
        weights_history = []

        current_weights = np.zeros(len(self.stock_returns.columns))

        for i in range(252, n_days):  # Start after 1 year of data
            if (i - 252) % rebalance_frequency == 0:
                # Calculate momentum scores
                lookback_returns = self.stock_returns.iloc[i - 252:i]
                mom_scores = lookback_returns.sum()

                # Select top N stocks
                top_stocks = mom_scores.nlargest(n_stocks).index
                new_weights = np.zeros(len(self.stock_returns.columns))

                for stock in top_stocks:
                    idx = list(self.stock_returns.columns).index(stock)
                    new_weights[idx] = 1 / n_stocks

                current_weights = new_weights
                weights_history.append(current_weights.copy())

            # Calculate portfolio return
            day_returns = self.stock_returns.iloc[i].values
            port_return = np.dot(current_weights, day_returns)
            portfolio_returns.append(port_return)

        portfolio_returns = np.array(portfolio_returns)
        cumulative_return = np.prod(1 + portfolio_returns) - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(portfolio_returns)) - 1
        annualized_vol = np.std(portfolio_returns) * np.sqrt(252)
        sharpe = (annualized_return - 0.02) / annualized_vol if annualized_vol > 0 else 0

        # Max drawdown
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        max_drawdown = np.max((running_max - cumulative) / running_max)

        return {
            'returns': portfolio_returns,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'initial_capital': initial_capital,
            'final_capital': initial_capital * (1 + cumulative_return)
        }
