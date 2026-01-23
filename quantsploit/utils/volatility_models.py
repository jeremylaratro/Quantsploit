"""
GARCH Volatility Modeling Module for Quantsploit

This module provides advanced volatility modeling capabilities including:
- GARCH(p,q) model fitting and forecasting
- EGARCH for asymmetric volatility effects
- GJR-GARCH for leverage effects
- Volatility regime detection and classification
- Dynamic VaR and stop-loss calculations
- Integration with the existing backtesting framework

The module attempts to use the 'arch' library for GARCH modeling,
falling back to EWMA (Exponentially Weighted Moving Average) volatility
estimation when arch is not available.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass
from enum import Enum
import warnings
import logging

logger = logging.getLogger(__name__)

# Attempt to import arch library for GARCH models
try:
    from arch import arch_model
    from arch.univariate import GARCH, EGARCH, ConstantMean, ZeroMean
    ARCH_AVAILABLE = True
    logger.info("arch library available - using full GARCH implementation")
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning(
        "arch library not installed. Using EWMA fallback for volatility estimation. "
        "Install with: pip install arch"
    )


class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class GARCHFitResult:
    """Results from fitting a GARCH model"""
    model_type: str
    p: int
    q: int
    params: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    conditional_volatility: pd.Series
    standardized_residuals: pd.Series
    converged: bool
    using_fallback: bool = False

    def summary(self) -> str:
        """Return a summary string of the fit results"""
        lines = [
            f"Model: {self.model_type}({self.p},{self.q})",
            f"Converged: {self.converged}",
            f"Log-Likelihood: {self.log_likelihood:.4f}",
            f"AIC: {self.aic:.4f}",
            f"BIC: {self.bic:.4f}",
            "",
            "Parameters:",
        ]
        for name, value in self.params.items():
            lines.append(f"  {name}: {value:.6f}")

        if self.using_fallback:
            lines.append("")
            lines.append("WARNING: Using EWMA fallback (arch library not available)")

        return "\n".join(lines)


@dataclass
class VolatilityForecast:
    """Results from volatility forecasting"""
    horizon: int
    variance_forecast: np.ndarray
    volatility_forecast: np.ndarray  # sqrt of variance
    mean_forecast: Optional[np.ndarray] = None
    confidence_intervals: Optional[Dict[str, np.ndarray]] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast to DataFrame"""
        data = {
            'variance': self.variance_forecast,
            'volatility': self.volatility_forecast,
        }
        if self.mean_forecast is not None:
            data['mean'] = self.mean_forecast
        if self.confidence_intervals:
            for key, values in self.confidence_intervals.items():
                data[key] = values
        return pd.DataFrame(data, index=range(1, self.horizon + 1))


class GARCHModel:
    """
    GARCH Volatility Model class supporting multiple model types.

    This class provides methods for fitting GARCH, EGARCH, and GJR-GARCH models
    to return series data, generating volatility forecasts, and extracting
    conditional volatility estimates.

    If the arch library is not available, falls back to EWMA volatility estimation.

    Example:
        >>> model = GARCHModel(returns_series)
        >>> result = model.fit_garch(p=1, q=1)
        >>> forecast = model.forecast_volatility(horizon=10)
        >>> cond_vol = model.get_conditional_volatility()
    """

    def __init__(
        self,
        returns: Union[pd.Series, np.ndarray],
        scale: float = 100.0
    ):
        """
        Initialize the GARCH model.

        Args:
            returns: Return series (can be price returns or log returns)
            scale: Scaling factor for returns (100 for percentage returns).
                   GARCH models work better with scaled data.
        """
        if isinstance(returns, pd.Series):
            self.returns = returns.dropna()
            self.index = self.returns.index
        else:
            self.returns = pd.Series(returns).dropna()
            self.index = self.returns.index

        self.scale = scale
        self.scaled_returns = self.returns * scale

        self._fitted_model = None
        self._fit_result: Optional[GARCHFitResult] = None
        self._model_type: Optional[str] = None

    def fit_garch(
        self,
        p: int = 1,
        q: int = 1,
        mean: str = 'constant',
        dist: str = 'normal',
        update_freq: int = 0
    ) -> GARCHFitResult:
        """
        Fit a GARCH(p,q) model to the return series.

        The GARCH(p,q) model:
            sigma^2_t = omega + sum_{i=1}^{p} alpha_i * epsilon^2_{t-i}
                        + sum_{j=1}^{q} beta_j * sigma^2_{t-j}

        Args:
            p: Order of the GARCH (lagged variance) component
            q: Order of the ARCH (lagged squared returns) component
            mean: Mean model specification ('constant', 'zero', 'ar')
            dist: Error distribution ('normal', 't', 'skewt', 'ged')
            update_freq: Frequency to update progress (0 = no updates)

        Returns:
            GARCHFitResult with model parameters and diagnostics
        """
        if ARCH_AVAILABLE:
            return self._fit_garch_arch(p, q, mean, dist, update_freq)
        else:
            return self._fit_garch_ewma_fallback(p, q)

    def fit_egarch(
        self,
        p: int = 1,
        q: int = 1,
        o: int = 1,
        mean: str = 'constant',
        dist: str = 'normal',
        update_freq: int = 0
    ) -> GARCHFitResult:
        """
        Fit an EGARCH(p,o,q) model to capture asymmetric volatility effects.

        The EGARCH model allows for asymmetric effects where negative shocks
        can have different impacts on volatility than positive shocks.

        log(sigma^2_t) = omega + sum_{i=1}^{p} beta_i * log(sigma^2_{t-i})
                         + sum_{j=1}^{o} gamma_j * z_{t-j}
                         + sum_{k=1}^{q} alpha_k * (|z_{t-k}| - E[|z_{t-k}|])

        Args:
            p: Order of the log-variance component
            q: Order of the absolute standardized residual component
            o: Order of the asymmetric term (leverage effect)
            mean: Mean model specification
            dist: Error distribution
            update_freq: Update frequency during optimization

        Returns:
            GARCHFitResult with model parameters and diagnostics
        """
        if ARCH_AVAILABLE:
            return self._fit_egarch_arch(p, q, o, mean, dist, update_freq)
        else:
            return self._fit_egarch_ewma_fallback(p, q, o)

    def fit_gjr_garch(
        self,
        p: int = 1,
        q: int = 1,
        o: int = 1,
        mean: str = 'constant',
        dist: str = 'normal',
        update_freq: int = 0
    ) -> GARCHFitResult:
        """
        Fit a GJR-GARCH(p,o,q) model to capture leverage effects.

        The GJR-GARCH model (also known as Threshold GARCH) allows for
        asymmetric effects where negative returns increase volatility more
        than positive returns of the same magnitude.

        sigma^2_t = omega + sum_{i=1}^{p} (alpha_i + gamma_i * I_{t-i}) * epsilon^2_{t-i}
                    + sum_{j=1}^{q} beta_j * sigma^2_{t-j}

        where I_{t-i} = 1 if epsilon_{t-i} < 0, else 0

        Args:
            p: Order of the ARCH component
            q: Order of the GARCH component
            o: Order of the asymmetric (leverage) component
            mean: Mean model specification
            dist: Error distribution
            update_freq: Update frequency during optimization

        Returns:
            GARCHFitResult with model parameters and diagnostics
        """
        if ARCH_AVAILABLE:
            return self._fit_gjr_garch_arch(p, q, o, mean, dist, update_freq)
        else:
            return self._fit_gjr_garch_ewma_fallback(p, q, o)

    def forecast_volatility(
        self,
        horizon: int = 10,
        method: str = 'analytic',
        simulations: int = 1000
    ) -> VolatilityForecast:
        """
        Generate N-step ahead volatility forecasts.

        Args:
            horizon: Number of periods to forecast
            method: Forecasting method ('analytic', 'simulation', 'bootstrap')
            simulations: Number of simulations for simulation-based methods

        Returns:
            VolatilityForecast with variance and volatility predictions

        Raises:
            ValueError: If model has not been fitted
        """
        if self._fitted_model is None:
            raise ValueError("Model must be fitted before forecasting. Call fit_garch() first.")

        if ARCH_AVAILABLE and not self._fit_result.using_fallback:
            return self._forecast_arch(horizon, method, simulations)
        else:
            return self._forecast_ewma_fallback(horizon)

    def get_conditional_volatility(self) -> pd.Series:
        """
        Get the historical conditional volatility series from the fitted model.

        Returns:
            pd.Series: Conditional volatility (annualized standard deviation)

        Raises:
            ValueError: If model has not been fitted
        """
        if self._fit_result is None:
            raise ValueError("Model must be fitted first. Call fit_garch() first.")

        # Return conditional volatility, unscaling if necessary
        cond_vol = self._fit_result.conditional_volatility / self.scale

        # Annualize (assuming daily data, 252 trading days)
        annualized_vol = cond_vol * np.sqrt(252)

        return annualized_vol

    def get_standardized_residuals(self) -> pd.Series:
        """
        Get standardized residuals from the fitted model.

        Standardized residuals should be approximately i.i.d. if the model
        is correctly specified.

        Returns:
            pd.Series: Standardized residuals
        """
        if self._fit_result is None:
            raise ValueError("Model must be fitted first.")

        return self._fit_result.standardized_residuals

    # ===== Internal arch library implementations =====

    def _fit_garch_arch(
        self, p: int, q: int, mean: str, dist: str, update_freq: int
    ) -> GARCHFitResult:
        """Fit GARCH using arch library"""
        model = arch_model(
            self.scaled_returns,
            mean=mean,
            vol='GARCH',
            p=p,
            q=q,
            dist=dist
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp='off', update_freq=update_freq)

        self._fitted_model = res
        self._model_type = 'GARCH'

        # Extract parameters
        params = dict(res.params)

        cond_vol = pd.Series(res.conditional_volatility, index=self.index)
        std_resid = pd.Series(res.std_resid, index=self.index)

        self._fit_result = GARCHFitResult(
            model_type='GARCH',
            p=p,
            q=q,
            params=params,
            log_likelihood=res.loglikelihood,
            aic=res.aic,
            bic=res.bic,
            conditional_volatility=cond_vol,
            standardized_residuals=std_resid,
            converged=res.convergence_flag == 0,
            using_fallback=False
        )

        return self._fit_result

    def _fit_egarch_arch(
        self, p: int, q: int, o: int, mean: str, dist: str, update_freq: int
    ) -> GARCHFitResult:
        """Fit EGARCH using arch library"""
        model = arch_model(
            self.scaled_returns,
            mean=mean,
            vol='EGARCH',
            p=p,
            q=q,
            o=o,
            dist=dist
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp='off', update_freq=update_freq)

        self._fitted_model = res
        self._model_type = 'EGARCH'

        params = dict(res.params)

        cond_vol = pd.Series(res.conditional_volatility, index=self.index)
        std_resid = pd.Series(res.std_resid, index=self.index)

        self._fit_result = GARCHFitResult(
            model_type='EGARCH',
            p=p,
            q=q,
            params=params,
            log_likelihood=res.loglikelihood,
            aic=res.aic,
            bic=res.bic,
            conditional_volatility=cond_vol,
            standardized_residuals=std_resid,
            converged=res.convergence_flag == 0,
            using_fallback=False
        )

        return self._fit_result

    def _fit_gjr_garch_arch(
        self, p: int, q: int, o: int, mean: str, dist: str, update_freq: int
    ) -> GARCHFitResult:
        """Fit GJR-GARCH using arch library"""
        model = arch_model(
            self.scaled_returns,
            mean=mean,
            vol='GARCH',
            p=p,
            o=o,
            q=q,
            dist=dist
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp='off', update_freq=update_freq)

        self._fitted_model = res
        self._model_type = 'GJR-GARCH'

        params = dict(res.params)

        cond_vol = pd.Series(res.conditional_volatility, index=self.index)
        std_resid = pd.Series(res.std_resid, index=self.index)

        self._fit_result = GARCHFitResult(
            model_type='GJR-GARCH',
            p=p,
            q=q,
            params=params,
            log_likelihood=res.loglikelihood,
            aic=res.aic,
            bic=res.bic,
            conditional_volatility=cond_vol,
            standardized_residuals=std_resid,
            converged=res.convergence_flag == 0,
            using_fallback=False
        )

        return self._fit_result

    def _forecast_arch(
        self, horizon: int, method: str, simulations: int
    ) -> VolatilityForecast:
        """Generate forecast using arch library"""
        if method == 'simulation':
            forecast = self._fitted_model.forecast(
                horizon=horizon,
                method='simulation',
                simulations=simulations
            )
        elif method == 'bootstrap':
            forecast = self._fitted_model.forecast(
                horizon=horizon,
                method='bootstrap',
                simulations=simulations
            )
        else:  # analytic
            forecast = self._fitted_model.forecast(horizon=horizon)

        # Extract variance forecast (last row contains the forecast)
        variance = forecast.variance.iloc[-1].values / (self.scale ** 2)
        volatility = np.sqrt(variance)

        # Get mean forecast if available
        mean_forecast = None
        if hasattr(forecast, 'mean') and forecast.mean is not None:
            mean_forecast = forecast.mean.iloc[-1].values / self.scale

        return VolatilityForecast(
            horizon=horizon,
            variance_forecast=variance,
            volatility_forecast=volatility,
            mean_forecast=mean_forecast
        )

    # ===== EWMA Fallback implementations =====

    def _fit_garch_ewma_fallback(self, p: int, q: int) -> GARCHFitResult:
        """
        Fallback GARCH estimation using EWMA when arch library is unavailable.

        EWMA volatility: sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_{t-1}
        This is equivalent to an integrated GARCH(1,1) model with specific constraints.
        """
        logger.warning(
            "Using EWMA fallback for GARCH estimation. "
            "Install 'arch' library for full GARCH functionality: pip install arch"
        )

        # Optimal lambda for EWMA (RiskMetrics uses 0.94 for daily data)
        # We can estimate optimal lambda via maximum likelihood
        optimal_lambda = self._estimate_optimal_ewma_lambda()

        # Calculate EWMA variance
        returns_sq = self.scaled_returns ** 2
        variance = self._calculate_ewma_variance(returns_sq, optimal_lambda)
        cond_vol = np.sqrt(variance)

        # Standardized residuals
        std_resid = self.scaled_returns / cond_vol

        # Approximate log-likelihood (assuming normal distribution)
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi) + np.log(variance) + (self.scaled_returns ** 2) / variance
        )

        # Approximate AIC/BIC (1 parameter: lambda)
        n = len(self.returns)
        k = 1
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + k * np.log(n)

        cond_vol_series = pd.Series(cond_vol, index=self.index)
        std_resid_series = pd.Series(std_resid, index=self.index)

        self._fitted_model = {'lambda': optimal_lambda, 'variance': variance}
        self._model_type = 'EWMA'

        self._fit_result = GARCHFitResult(
            model_type='GARCH (EWMA Fallback)',
            p=p,
            q=q,
            params={
                'omega': 0.0,  # EWMA has no constant
                'alpha[1]': 1 - optimal_lambda,
                'beta[1]': optimal_lambda,
                'lambda': optimal_lambda
            },
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            conditional_volatility=cond_vol_series,
            standardized_residuals=std_resid_series,
            converged=True,
            using_fallback=True
        )

        return self._fit_result

    def _fit_egarch_ewma_fallback(self, p: int, q: int, o: int) -> GARCHFitResult:
        """
        Fallback EGARCH estimation using asymmetric EWMA.

        Incorporates a simple asymmetric term: larger weight on negative returns.
        """
        logger.warning(
            "Using asymmetric EWMA fallback for EGARCH estimation. "
            "Install 'arch' library for true EGARCH: pip install arch"
        )

        optimal_lambda = self._estimate_optimal_ewma_lambda()
        asymmetry_factor = 1.5  # Negative shocks have 50% more impact

        # Calculate asymmetric EWMA variance
        variance = self._calculate_asymmetric_ewma_variance(
            self.scaled_returns, optimal_lambda, asymmetry_factor
        )
        cond_vol = np.sqrt(variance)

        # Standardized residuals
        std_resid = self.scaled_returns / cond_vol

        # Approximate log-likelihood
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi) + np.log(variance) + (self.scaled_returns ** 2) / variance
        )

        n = len(self.returns)
        k = 2  # lambda and asymmetry
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + k * np.log(n)

        cond_vol_series = pd.Series(cond_vol, index=self.index)
        std_resid_series = pd.Series(std_resid, index=self.index)

        self._fitted_model = {
            'lambda': optimal_lambda,
            'asymmetry': asymmetry_factor,
            'variance': variance
        }
        self._model_type = 'Asymmetric EWMA'

        self._fit_result = GARCHFitResult(
            model_type='EGARCH (Asymmetric EWMA Fallback)',
            p=p,
            q=q,
            params={
                'lambda': optimal_lambda,
                'asymmetry': asymmetry_factor,
                'alpha': 1 - optimal_lambda,
                'beta': optimal_lambda
            },
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            conditional_volatility=cond_vol_series,
            standardized_residuals=std_resid_series,
            converged=True,
            using_fallback=True
        )

        return self._fit_result

    def _fit_gjr_garch_ewma_fallback(self, p: int, q: int, o: int) -> GARCHFitResult:
        """
        Fallback GJR-GARCH estimation using threshold EWMA.

        Similar to asymmetric EWMA but uses indicator function approach.
        """
        logger.warning(
            "Using threshold EWMA fallback for GJR-GARCH estimation. "
            "Install 'arch' library for true GJR-GARCH: pip install arch"
        )

        optimal_lambda = self._estimate_optimal_ewma_lambda()
        gamma = 0.1  # Leverage coefficient for negative returns

        # Calculate threshold EWMA variance
        variance = self._calculate_threshold_ewma_variance(
            self.scaled_returns, optimal_lambda, gamma
        )
        cond_vol = np.sqrt(variance)

        # Standardized residuals
        std_resid = self.scaled_returns / cond_vol

        # Approximate log-likelihood
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi) + np.log(variance) + (self.scaled_returns ** 2) / variance
        )

        n = len(self.returns)
        k = 2  # lambda and gamma
        aic = -2 * log_likelihood + 2 * k
        bic = -2 * log_likelihood + k * np.log(n)

        cond_vol_series = pd.Series(cond_vol, index=self.index)
        std_resid_series = pd.Series(std_resid, index=self.index)

        self._fitted_model = {
            'lambda': optimal_lambda,
            'gamma': gamma,
            'variance': variance
        }
        self._model_type = 'Threshold EWMA'

        self._fit_result = GARCHFitResult(
            model_type='GJR-GARCH (Threshold EWMA Fallback)',
            p=p,
            q=q,
            params={
                'lambda': optimal_lambda,
                'gamma': gamma,
                'alpha': 1 - optimal_lambda,
                'beta': optimal_lambda
            },
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            conditional_volatility=cond_vol_series,
            standardized_residuals=std_resid_series,
            converged=True,
            using_fallback=True
        )

        return self._fit_result

    def _forecast_ewma_fallback(self, horizon: int) -> VolatilityForecast:
        """Generate forecast using EWMA when arch is unavailable"""
        if self._fitted_model is None:
            raise ValueError("Model must be fitted first.")

        variance = self._fitted_model['variance']
        last_variance = variance.iloc[-1] if isinstance(variance, pd.Series) else variance[-1]

        # For EWMA, multi-step forecasts converge to unconditional variance
        # But for simplicity, we use a persistence-based forecast
        lambda_param = self._fitted_model.get('lambda', 0.94)

        # Forecast variance: h-step ahead
        # For IGARCH (integrated GARCH like EWMA), variance forecast is constant
        variance_forecast = np.full(horizon, last_variance)

        # Unscale
        variance_forecast = variance_forecast / (self.scale ** 2)
        volatility_forecast = np.sqrt(variance_forecast)

        return VolatilityForecast(
            horizon=horizon,
            variance_forecast=variance_forecast,
            volatility_forecast=volatility_forecast
        )

    def _estimate_optimal_ewma_lambda(self, grid_points: int = 50) -> float:
        """
        Estimate optimal lambda for EWMA via grid search maximizing likelihood.
        """
        lambdas = np.linspace(0.85, 0.99, grid_points)
        returns_sq = self.scaled_returns ** 2

        best_lambda = 0.94  # Default RiskMetrics
        best_ll = -np.inf

        for lam in lambdas:
            variance = self._calculate_ewma_variance(returns_sq, lam)
            # Avoid division by zero
            variance = np.maximum(variance, 1e-10)
            # Log-likelihood (simplified, assuming normality)
            ll = -0.5 * np.sum(np.log(variance) + returns_sq / variance)

            if ll > best_ll:
                best_ll = ll
                best_lambda = lam

        return best_lambda

    def _calculate_ewma_variance(
        self, returns_sq: pd.Series, lambda_param: float
    ) -> pd.Series:
        """Calculate EWMA variance series"""
        n = len(returns_sq)
        variance = np.zeros(n)

        # Initialize with sample variance
        variance[0] = returns_sq.iloc[:20].mean() if len(returns_sq) > 20 else returns_sq.mean()

        for t in range(1, n):
            variance[t] = lambda_param * variance[t-1] + (1 - lambda_param) * returns_sq.iloc[t-1]

        return pd.Series(variance, index=returns_sq.index)

    def _calculate_asymmetric_ewma_variance(
        self, returns: pd.Series, lambda_param: float, asymmetry: float
    ) -> pd.Series:
        """Calculate asymmetric EWMA variance (EGARCH-like fallback)"""
        n = len(returns)
        variance = np.zeros(n)

        # Initialize
        variance[0] = (returns ** 2).iloc[:20].mean() if len(returns) > 20 else (returns ** 2).mean()

        for t in range(1, n):
            r_prev = returns.iloc[t-1]
            # Apply asymmetry: negative returns have larger impact
            weight = asymmetry if r_prev < 0 else 1.0
            innovation = weight * (r_prev ** 2)
            variance[t] = lambda_param * variance[t-1] + (1 - lambda_param) * innovation

        return pd.Series(variance, index=returns.index)

    def _calculate_threshold_ewma_variance(
        self, returns: pd.Series, lambda_param: float, gamma: float
    ) -> pd.Series:
        """Calculate threshold EWMA variance (GJR-GARCH-like fallback)"""
        n = len(returns)
        variance = np.zeros(n)

        # Initialize
        variance[0] = (returns ** 2).iloc[:20].mean() if len(returns) > 20 else (returns ** 2).mean()

        alpha = 1 - lambda_param

        for t in range(1, n):
            r_prev = returns.iloc[t-1]
            indicator = 1.0 if r_prev < 0 else 0.0
            # GJR-style: extra gamma term for negative returns
            innovation = r_prev ** 2
            leverage_term = gamma * indicator * innovation
            variance[t] = (lambda_param * variance[t-1] +
                          alpha * innovation +
                          leverage_term)

        return pd.Series(variance, index=returns.index)


class VolatilityRegimeDetector:
    """
    Volatility Regime Detection and Classification.

    This class provides methods to classify market conditions into volatility
    regimes (Low, Medium, High, Extreme) based on percentile-based thresholds.

    The regime classification can be used for:
    - Dynamic position sizing (reduce exposure in high-vol regimes)
    - Strategy selection (mean reversion works better in low-vol)
    - Risk management (tighter stops in high-vol environments)

    Example:
        >>> detector = VolatilityRegimeDetector(returns_series)
        >>> regime = detector.detect_regime()
        >>> position_size = detector.regime_adjusted_position_size(base_size=1.0)
    """

    def __init__(
        self,
        returns: Union[pd.Series, np.ndarray],
        lookback_window: int = 252,
        volatility_window: int = 20
    ):
        """
        Initialize the regime detector.

        Args:
            returns: Return series for analysis
            lookback_window: Historical window for regime threshold calculation
            volatility_window: Rolling window for current volatility estimation
        """
        if isinstance(returns, pd.Series):
            self.returns = returns.dropna()
        else:
            self.returns = pd.Series(returns).dropna()

        self.lookback_window = lookback_window
        self.volatility_window = volatility_window

        # Calculate rolling volatility
        self.rolling_volatility = self.returns.rolling(
            window=volatility_window
        ).std() * np.sqrt(252)  # Annualized

        # Get regime thresholds
        self._thresholds = self._calculate_thresholds()

    def _calculate_thresholds(self) -> Dict[str, float]:
        """Calculate percentile-based regime thresholds"""
        vol = self.rolling_volatility.dropna()

        if len(vol) < self.lookback_window:
            # Use all available data
            reference_vol = vol
        else:
            reference_vol = vol.iloc[-self.lookback_window:]

        return {
            'low_medium': np.percentile(reference_vol, 33),
            'medium_high': np.percentile(reference_vol, 67),
            'high_extreme': np.percentile(reference_vol, 95)
        }

    def get_regime_thresholds(self) -> Dict[str, float]:
        """
        Get the current regime classification thresholds.

        Returns:
            Dictionary with threshold values:
            - low_medium: Threshold between Low and Medium regimes
            - medium_high: Threshold between Medium and High regimes
            - high_extreme: Threshold between High and Extreme regimes
        """
        return self._thresholds.copy()

    def detect_regime(
        self,
        current_volatility: Optional[float] = None
    ) -> VolatilityRegime:
        """
        Detect the current volatility regime.

        Args:
            current_volatility: Override current volatility value.
                               If None, uses the latest rolling volatility.

        Returns:
            VolatilityRegime enum value (LOW, MEDIUM, HIGH, EXTREME)
        """
        if current_volatility is None:
            current_volatility = self.rolling_volatility.iloc[-1]

        if pd.isna(current_volatility):
            return VolatilityRegime.MEDIUM  # Default

        if current_volatility <= self._thresholds['low_medium']:
            return VolatilityRegime.LOW
        elif current_volatility <= self._thresholds['medium_high']:
            return VolatilityRegime.MEDIUM
        elif current_volatility <= self._thresholds['high_extreme']:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME

    def detect_regime_series(self) -> pd.Series:
        """
        Detect volatility regime for the entire series.

        Returns:
            pd.Series of VolatilityRegime values
        """
        regimes = []
        for vol in self.rolling_volatility:
            if pd.isna(vol):
                regimes.append(None)
            else:
                regimes.append(self.detect_regime(vol))

        return pd.Series(regimes, index=self.returns.index)

    def regime_adjusted_position_size(
        self,
        base_size: float = 1.0,
        regime: Optional[VolatilityRegime] = None,
        scaling_factors: Optional[Dict[VolatilityRegime, float]] = None
    ) -> float:
        """
        Calculate position size adjusted for current volatility regime.

        In high volatility regimes, position sizes are reduced to maintain
        consistent risk exposure.

        Args:
            base_size: Base position size (e.g., 1.0 = 100% of capital allocation)
            regime: Override regime. If None, detects current regime.
            scaling_factors: Custom scaling factors per regime.
                            Default: LOW=1.2, MEDIUM=1.0, HIGH=0.6, EXTREME=0.3

        Returns:
            Adjusted position size
        """
        if scaling_factors is None:
            scaling_factors = {
                VolatilityRegime.LOW: 1.2,      # Can take slightly larger positions
                VolatilityRegime.MEDIUM: 1.0,   # Normal position size
                VolatilityRegime.HIGH: 0.6,     # Reduce exposure
                VolatilityRegime.EXTREME: 0.3   # Significantly reduce exposure
            }

        if regime is None:
            regime = self.detect_regime()

        scale = scaling_factors.get(regime, 1.0)
        return base_size * scale

    def get_regime_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for each volatility regime.

        Returns:
            Dictionary with statistics per regime including:
            - count: Number of observations
            - percentage: Percentage of time in regime
            - avg_return: Average return in regime
            - avg_volatility: Average volatility in regime
        """
        regime_series = self.detect_regime_series()

        stats = {}
        total_obs = regime_series.notna().sum()

        for regime in VolatilityRegime:
            mask = regime_series == regime
            count = mask.sum()

            if count > 0:
                regime_returns = self.returns[mask]
                regime_vol = self.rolling_volatility[mask]

                stats[regime.value] = {
                    'count': count,
                    'percentage': (count / total_obs) * 100,
                    'avg_return': regime_returns.mean() * 252,  # Annualized
                    'avg_volatility': regime_vol.mean(),
                    'sharpe': (regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                              if regime_returns.std() > 0 else 0)
                }
            else:
                stats[regime.value] = {
                    'count': 0,
                    'percentage': 0,
                    'avg_return': 0,
                    'avg_volatility': 0,
                    'sharpe': 0
                }

        return stats


# ===== Integration Functions =====

def calculate_dynamic_var(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.95,
    horizon: int = 1,
    model_type: str = 'garch',
    garch_p: int = 1,
    garch_q: int = 1
) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) using GARCH volatility forecasting.

    Dynamic VaR adapts to current market conditions by using conditional
    volatility estimates rather than historical volatility.

    Args:
        returns: Return series
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
        horizon: Forecast horizon in days
        model_type: 'garch', 'egarch', or 'gjr_garch'
        garch_p: GARCH p parameter
        garch_q: GARCH q parameter

    Returns:
        Tuple of (VaR, Expected Shortfall/CVaR)

    Example:
        >>> var, cvar = calculate_dynamic_var(returns, confidence_level=0.99)
        >>> print(f"99% VaR: {var:.4f}, CVaR: {cvar:.4f}")
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()

    # Fit GARCH model
    model = GARCHModel(returns)

    if model_type == 'egarch':
        result = model.fit_egarch(p=garch_p, q=garch_q)
    elif model_type == 'gjr_garch':
        result = model.fit_gjr_garch(p=garch_p, q=garch_q)
    else:
        result = model.fit_garch(p=garch_p, q=garch_q)

    # Get volatility forecast
    forecast = model.forecast_volatility(horizon=horizon)
    forecast_vol = forecast.volatility_forecast[0]

    # Calculate VaR (assuming normal distribution)
    from scipy import stats as scipy_stats
    z_score = scipy_stats.norm.ppf(1 - confidence_level)
    var = -z_score * forecast_vol

    # Calculate Expected Shortfall (CVaR)
    # ES = -E[R | R < -VaR] = sigma * phi(z) / (1 - alpha)
    phi_z = scipy_stats.norm.pdf(z_score)
    cvar = forecast_vol * phi_z / (1 - confidence_level)

    return var, cvar


def calculate_dynamic_stop_loss(
    prices: pd.Series,
    returns: pd.Series,
    multiplier: float = 2.0,
    model_type: str = 'garch',
    garch_p: int = 1,
    garch_q: int = 1,
    min_stop_pct: float = 0.01,
    max_stop_pct: float = 0.10
) -> Tuple[float, float]:
    """
    Calculate dynamic stop-loss levels using GARCH conditional volatility.

    Similar to ATR-based stops but uses GARCH volatility for more adaptive
    stop-loss levels that respond to changing market conditions.

    Args:
        prices: Price series
        returns: Return series (should correspond to prices)
        multiplier: Volatility multiplier for stop distance (default 2.0)
        model_type: 'garch', 'egarch', or 'gjr_garch'
        garch_p: GARCH p parameter
        garch_q: GARCH q parameter
        min_stop_pct: Minimum stop-loss percentage
        max_stop_pct: Maximum stop-loss percentage

    Returns:
        Tuple of (stop_loss_price, stop_distance_percent)

    Example:
        >>> stop_price, stop_pct = calculate_dynamic_stop_loss(prices, returns)
        >>> print(f"Stop at ${stop_price:.2f} ({stop_pct:.2%} below current)")
    """
    if isinstance(returns, pd.Series):
        returns = returns.dropna()

    current_price = prices.iloc[-1]

    # Fit GARCH model
    model = GARCHModel(returns)

    if model_type == 'egarch':
        model.fit_egarch(p=garch_p, q=garch_q)
    elif model_type == 'gjr_garch':
        model.fit_gjr_garch(p=garch_p, q=garch_q)
    else:
        model.fit_garch(p=garch_p, q=garch_q)

    # Get current conditional volatility (daily)
    cond_vol = model.get_conditional_volatility()
    current_vol = cond_vol.iloc[-1] / np.sqrt(252)  # Convert to daily

    # Calculate stop distance
    stop_distance_pct = multiplier * current_vol

    # Apply min/max constraints
    stop_distance_pct = np.clip(stop_distance_pct, min_stop_pct, max_stop_pct)

    # Calculate stop price (for long position)
    stop_price = current_price * (1 - stop_distance_pct)

    return stop_price, stop_distance_pct


def calculate_garch_adjusted_position_size(
    returns: pd.Series,
    target_volatility: float = 0.15,
    max_leverage: float = 2.0,
    model_type: str = 'garch'
) -> float:
    """
    Calculate position size to achieve target portfolio volatility.

    Uses GARCH volatility forecasts to dynamically adjust position sizes
    to maintain consistent risk exposure.

    Args:
        returns: Return series
        target_volatility: Target annualized volatility (e.g., 0.15 = 15%)
        max_leverage: Maximum allowed leverage
        model_type: GARCH model type

    Returns:
        Position size multiplier (1.0 = 100% allocation)

    Example:
        >>> size = calculate_garch_adjusted_position_size(returns, target_vol=0.10)
        >>> print(f"Allocate {size*100:.1f}% of capital")
    """
    model = GARCHModel(returns)

    if model_type == 'egarch':
        model.fit_egarch()
    elif model_type == 'gjr_garch':
        model.fit_gjr_garch()
    else:
        model.fit_garch()

    # Get current annualized volatility
    cond_vol = model.get_conditional_volatility()
    current_vol = cond_vol.iloc[-1]

    if current_vol <= 0 or pd.isna(current_vol):
        return 1.0

    # Calculate position size to achieve target vol
    position_size = target_volatility / current_vol

    # Apply leverage constraint
    position_size = min(position_size, max_leverage)

    # Don't go below some minimum
    position_size = max(position_size, 0.1)

    return position_size


# ===== Backtesting Integration =====

class GARCHBacktestIntegration:
    """
    Integration class for using GARCH models within the Quantsploit backtesting framework.

    This class provides methods that can be used within backtest strategies
    to make volatility-aware trading decisions.

    Example:
        >>> integration = GARCHBacktestIntegration(lookback=252)
        >>> # Within backtest strategy:
        >>> regime = integration.get_current_regime(history)
        >>> stop_price = integration.get_dynamic_stop(history, entry_price)
    """

    def __init__(
        self,
        lookback: int = 252,
        garch_p: int = 1,
        garch_q: int = 1,
        model_type: str = 'garch',
        refit_frequency: int = 20
    ):
        """
        Initialize GARCH backtest integration.

        Args:
            lookback: Historical lookback for model fitting
            garch_p: GARCH p parameter
            garch_q: GARCH q parameter
            model_type: Type of GARCH model
            refit_frequency: How often to refit the model (in bars)
        """
        self.lookback = lookback
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.model_type = model_type
        self.refit_frequency = refit_frequency

        self._last_fit_bar = -refit_frequency
        self._cached_model: Optional[GARCHModel] = None
        self._cached_regime_detector: Optional[VolatilityRegimeDetector] = None

    def update(self, data: pd.DataFrame, current_bar: int) -> None:
        """
        Update models if needed based on refit frequency.

        Args:
            data: Full historical data DataFrame with 'Close' column
            current_bar: Current bar index
        """
        if current_bar - self._last_fit_bar >= self.refit_frequency:
            self._refit_models(data, current_bar)
            self._last_fit_bar = current_bar

    def _refit_models(self, data: pd.DataFrame, current_bar: int) -> None:
        """Refit GARCH and regime detection models"""
        # Get returns
        history = data.iloc[max(0, current_bar - self.lookback):current_bar + 1]
        if len(history) < 50:
            return

        returns = history['Close'].pct_change().dropna()

        if len(returns) < 50:
            return

        try:
            # Fit GARCH model
            self._cached_model = GARCHModel(returns)
            if self.model_type == 'egarch':
                self._cached_model.fit_egarch(p=self.garch_p, q=self.garch_q)
            elif self.model_type == 'gjr_garch':
                self._cached_model.fit_gjr_garch(p=self.garch_p, q=self.garch_q)
            else:
                self._cached_model.fit_garch(p=self.garch_p, q=self.garch_q)

            # Update regime detector
            self._cached_regime_detector = VolatilityRegimeDetector(returns)
        except Exception as e:
            logger.warning(f"Failed to fit GARCH model: {e}")

    def get_current_regime(self, data: pd.DataFrame, current_bar: int) -> VolatilityRegime:
        """
        Get current volatility regime.

        Args:
            data: Historical data
            current_bar: Current bar index

        Returns:
            Current VolatilityRegime
        """
        self.update(data, current_bar)

        if self._cached_regime_detector is None:
            return VolatilityRegime.MEDIUM

        return self._cached_regime_detector.detect_regime()

    def get_position_size_multiplier(
        self,
        data: pd.DataFrame,
        current_bar: int,
        base_size: float = 1.0
    ) -> float:
        """
        Get regime-adjusted position size multiplier.

        Args:
            data: Historical data
            current_bar: Current bar index
            base_size: Base position size

        Returns:
            Adjusted position size
        """
        self.update(data, current_bar)

        if self._cached_regime_detector is None:
            return base_size

        return self._cached_regime_detector.regime_adjusted_position_size(base_size)

    def get_dynamic_stop_loss(
        self,
        data: pd.DataFrame,
        current_bar: int,
        entry_price: float,
        multiplier: float = 2.0
    ) -> float:
        """
        Get dynamic stop-loss price based on GARCH volatility.

        Args:
            data: Historical data
            current_bar: Current bar index
            entry_price: Entry price for the position
            multiplier: Volatility multiplier

        Returns:
            Stop-loss price
        """
        self.update(data, current_bar)

        if self._cached_model is None:
            # Fallback: use simple percentage stop
            return entry_price * 0.95

        try:
            cond_vol = self._cached_model.get_conditional_volatility()
            daily_vol = cond_vol.iloc[-1] / np.sqrt(252)

            stop_distance = multiplier * daily_vol
            stop_distance = np.clip(stop_distance, 0.01, 0.10)

            return entry_price * (1 - stop_distance)
        except Exception:
            return entry_price * 0.95

    def get_volatility_forecast(
        self,
        data: pd.DataFrame,
        current_bar: int,
        horizon: int = 5
    ) -> Optional[np.ndarray]:
        """
        Get volatility forecast.

        Args:
            data: Historical data
            current_bar: Current bar index
            horizon: Forecast horizon

        Returns:
            Array of forecasted volatilities or None
        """
        self.update(data, current_bar)

        if self._cached_model is None:
            return None

        try:
            forecast = self._cached_model.forecast_volatility(horizon=horizon)
            return forecast.volatility_forecast
        except Exception:
            return None


# ===== Example Usage =====

def example_usage():
    """
    Example demonstrating GARCH volatility modeling usage.
    """
    import numpy as np

    print("=" * 60)
    print("GARCH Volatility Modeling - Example Usage")
    print("=" * 60)

    # Generate sample data (simulated returns)
    np.random.seed(42)
    n = 500

    # Simulate GARCH(1,1) process for realistic returns
    omega = 0.00001
    alpha = 0.1
    beta = 0.85

    returns = np.zeros(n)
    variance = np.zeros(n)
    variance[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
        returns[t] = np.sqrt(variance[t]) * np.random.standard_normal()

    returns_series = pd.Series(returns, index=pd.date_range('2022-01-01', periods=n, freq='D'))

    print("\n1. Fitting GARCH(1,1) Model")
    print("-" * 40)

    model = GARCHModel(returns_series)
    result = model.fit_garch(p=1, q=1)
    print(result.summary())

    print("\n2. Volatility Forecasting")
    print("-" * 40)

    forecast = model.forecast_volatility(horizon=10)
    print(f"10-day volatility forecast:")
    print(forecast.to_dataframe())

    print("\n3. Regime Detection")
    print("-" * 40)

    detector = VolatilityRegimeDetector(returns_series)
    current_regime = detector.detect_regime()
    print(f"Current regime: {current_regime.value}")
    print(f"\nRegime thresholds: {detector.get_regime_thresholds()}")

    print("\n4. Regime Statistics")
    print("-" * 40)

    stats = detector.get_regime_statistics()
    for regime, stat in stats.items():
        print(f"\n{regime.upper()}:")
        print(f"  Count: {stat['count']}")
        print(f"  Percentage: {stat['percentage']:.1f}%")
        print(f"  Avg Return: {stat['avg_return']:.4f}")
        print(f"  Avg Vol: {stat['avg_volatility']:.4f}")

    print("\n5. Dynamic Position Sizing")
    print("-" * 40)

    base_size = 1.0
    adjusted_size = detector.regime_adjusted_position_size(base_size)
    print(f"Base position size: {base_size}")
    print(f"Regime-adjusted size: {adjusted_size:.2f}")

    print("\n6. Dynamic VaR Calculation")
    print("-" * 40)

    try:
        var_95, cvar_95 = calculate_dynamic_var(returns_series, confidence_level=0.95)
        print(f"95% VaR: {var_95:.4f}")
        print(f"95% CVaR: {cvar_95:.4f}")
    except ImportError:
        print("scipy not available for VaR calculation")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
