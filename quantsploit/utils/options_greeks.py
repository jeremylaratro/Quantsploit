"""
Options Greeks Calculator

This module provides functions to calculate options Greeks using
the Black-Scholes model and other pricing models.

First-Order Greeks:
- Delta: Rate of change of option price with respect to underlying price
- Gamma: Rate of change of delta with respect to underlying price
- Theta: Rate of change of option price with respect to time
- Vega: Rate of change of option price with respect to volatility
- Rho: Rate of change of option price with respect to interest rate

Second-Order Greeks (Advanced):
- Vanna: Sensitivity of delta to volatility (∂Δ/∂σ = ∂Vega/∂S)
- Volga (Vomma): Sensitivity of vega to volatility (∂²V/∂σ²)
- Charm (Delta Decay): Sensitivity of delta to time (∂Δ/∂t)
- Veta: Sensitivity of vega to time (∂Vega/∂t)
- Speed: Third derivative of price with respect to spot (∂Γ/∂S)
- Zomma: Sensitivity of gamma to volatility (∂Γ/∂σ)
- Color: Sensitivity of gamma to time (∂Γ/∂t)
- Ultima: Sensitivity of volga to volatility (∂³V/∂σ³)

Additional calculations:
- Implied Volatility using Newton-Raphson method
- Black-Scholes option pricing
- Probability of profit
- American options pricing (Binomial Tree - CRR model)
- Implied Volatility Surface (SVI parameterization)
- Options Risk Dashboard
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, brentq
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Callable
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Data Classes and Enums
# =============================================================================

class OptionType(Enum):
    """Option type enumeration"""
    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    """Option exercise style"""
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class GreeksResult:
    """Complete Greeks result with first and second order"""
    # First-order Greeks
    delta: float
    gamma: float
    theta: float  # Daily
    vega: float   # Per 1% vol change
    rho: float    # Per 1% rate change

    # Second-order Greeks
    vanna: float = 0.0      # ∂Δ/∂σ
    volga: float = 0.0      # ∂²V/∂σ² (Vomma)
    charm: float = 0.0      # ∂Δ/∂t (Delta decay)
    veta: float = 0.0       # ∂Vega/∂t
    speed: float = 0.0      # ∂Γ/∂S
    zomma: float = 0.0      # ∂Γ/∂σ
    color: float = 0.0      # ∂Γ/∂t
    ultima: float = 0.0     # ∂³V/∂σ³

    # Option metrics
    price: float = 0.0
    intrinsic_value: float = 0.0
    extrinsic_value: float = 0.0


@dataclass
class SVIParams:
    """Stochastic Volatility Inspired (SVI) surface parameters

    SVI parameterization: w(k) = a + b*(ρ*(k-m) + sqrt((k-m)² + σ²))
    where k = log(K/F) is log-moneyness
    """
    a: float  # Overall variance level
    b: float  # Slope of wings (ATM volatility of volatility)
    rho: float  # Correlation determining skew direction
    m: float  # Center of the smile (ATM shift)
    sigma: float  # Width of the smile's minimum variance region

    def total_variance(self, k: float) -> float:
        """Calculate total variance w(k) for log-moneyness k"""
        return self.a + self.b * (
            self.rho * (k - self.m) +
            np.sqrt((k - self.m)**2 + self.sigma**2)
        )

    def implied_vol(self, k: float, T: float) -> float:
        """Calculate implied volatility from total variance"""
        w = self.total_variance(k)
        return np.sqrt(max(w / T, 1e-10))


@dataclass
class IVSurfacePoint:
    """Single point on IV surface"""
    strike: float
    expiry: float  # Years
    implied_vol: float
    market_price: float
    model_price: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class IVSurface:
    """Implied Volatility Surface"""
    spot: float
    rate: float
    dividend_yield: float
    points: List[IVSurfacePoint] = field(default_factory=list)
    svi_params: Dict[float, SVIParams] = field(default_factory=dict)  # expiry -> params

    def get_iv(self, strike: float, expiry: float) -> Optional[float]:
        """Interpolate IV for given strike and expiry"""
        if expiry in self.svi_params:
            forward = self.spot * np.exp((self.rate - self.dividend_yield) * expiry)
            k = np.log(strike / forward)
            return self.svi_params[expiry].implied_vol(k, expiry)

        # Linear interpolation fallback
        expiries = sorted(self.svi_params.keys())
        if not expiries:
            return None

        if expiry <= expiries[0]:
            forward = self.spot * np.exp((self.rate - self.dividend_yield) * expiries[0])
            k = np.log(strike / forward)
            return self.svi_params[expiries[0]].implied_vol(k, expiries[0])

        if expiry >= expiries[-1]:
            forward = self.spot * np.exp((self.rate - self.dividend_yield) * expiries[-1])
            k = np.log(strike / forward)
            return self.svi_params[expiries[-1]].implied_vol(k, expiries[-1])

        # Interpolate between two expiries in VARIANCE SPACE (not IV space)
        # Linear interpolation of IV causes calendar arbitrage because variance
        # must be monotonically increasing in time. Interpolating in variance
        # space (w = σ² * T) preserves this constraint.
        for i in range(len(expiries) - 1):
            if expiries[i] <= expiry < expiries[i+1]:
                t1, t2 = expiries[i], expiries[i+1]
                weight = (expiry - t1) / (t2 - t1)

                forward1 = self.spot * np.exp((self.rate - self.dividend_yield) * t1)
                forward2 = self.spot * np.exp((self.rate - self.dividend_yield) * t2)

                k1 = np.log(strike / forward1)
                k2 = np.log(strike / forward2)

                iv1 = self.svi_params[t1].implied_vol(k1, t1)
                iv2 = self.svi_params[t2].implied_vol(k2, t2)

                # Convert to total variance space for interpolation
                w1 = (iv1 ** 2) * t1
                w2 = (iv2 ** 2) * t2
                w_interp = (1 - weight) * w1 + weight * w2

                # Convert back to IV
                return np.sqrt(w_interp / expiry)

        return None


@dataclass
class OptionsRiskReport:
    """Comprehensive options risk report"""
    # Position summary
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0

    # Dollar-denominated risks
    delta_dollars: float = 0.0  # $ P&L per $1 move
    gamma_dollars: float = 0.0  # $ Gamma per $1 move
    vega_dollars: float = 0.0   # $ P&L per 1% IV move
    theta_dollars: float = 0.0  # $ time decay per day

    # Second-order risks
    total_vanna: float = 0.0
    total_volga: float = 0.0
    total_charm: float = 0.0

    # Scenario analysis
    pnl_up_1pct: float = 0.0    # P&L if spot +1%
    pnl_down_1pct: float = 0.0  # P&L if spot -1%
    pnl_vol_up: float = 0.0     # P&L if vol +1%
    pnl_vol_down: float = 0.0   # P&L if vol -1%
    pnl_1day: float = 0.0       # P&L from 1 day decay

    # Risk limits
    max_loss_estimate: float = 0.0
    var_95: float = 0.0  # 95% VaR

    # Position details
    positions: List[Dict] = field(default_factory=list)


class OptionsGreeks:
    """Options Greeks calculator using Black-Scholes model"""

    @staticmethod
    def black_scholes_call(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate Black-Scholes call option price

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate (annual)
            sigma: Volatility (annual)
            q: Dividend yield (annual)

        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        return call_price

    @staticmethod
    def black_scholes_put(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate Black-Scholes put option price

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate (annual)
            sigma: Volatility (annual)
            q: Dividend yield (annual)

        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return put_price

    @staticmethod
    def delta_call(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate call option delta

        Delta measures the rate of change of option price with respect to
        changes in the underlying asset's price.

        Range: 0 to 1 for calls
        """
        if T <= 0:
            return 1.0 if S > K else 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        delta = np.exp(-q * T) * norm.cdf(d1)

        return delta

    @staticmethod
    def delta_put(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate put option delta

        Range: -1 to 0 for puts
        """
        if T <= 0:
            return -1.0 if S < K else 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        delta = -np.exp(-q * T) * norm.cdf(-d1)

        return delta

    @staticmethod
    def gamma(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate option gamma

        Gamma measures the rate of change of delta with respect to
        changes in the underlying price. Same for both calls and puts.

        Range: 0 to infinity
        Peaks at ATM
        """
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

        return gamma

    @staticmethod
    def theta_call(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate call option theta

        Theta measures the rate of change of option price with respect to
        time (time decay). Typically expressed as daily decay.

        Range: Negative for long options (time decay)
        """
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1))

        # Convert to daily theta (divide by 365)
        theta_daily = theta / 365

        return theta_daily

    @staticmethod
    def theta_put(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate put option theta (daily)

        Range: Negative for long options
        """
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))

        # Convert to daily theta
        theta_daily = theta / 365

        return theta_daily

    @staticmethod
    def vega(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate option vega

        Vega measures the rate of change of option price with respect to
        changes in volatility. Same for both calls and puts.

        Typically expressed as change in option price for 1% change in volatility.

        Range: 0 to infinity
        Peaks at ATM
        """
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% change

        return vega

    @staticmethod
    def rho_call(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate call option rho

        Rho measures the rate of change of option price with respect to
        changes in interest rate.

        Typically expressed as change in option price for 1% change in interest rate.

        Range: Positive for calls
        """
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Divide by 100 for 1% change

        return rho

    @staticmethod
    def rho_put(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate put option rho

        Range: Negative for puts
        """
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return rho

    # =========================================================================
    # Second-Order Greeks
    # =========================================================================

    @staticmethod
    def vanna(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate Vanna (DdeltaDvol or DvegaDspot)

        Vanna = ∂Δ/∂σ = ∂Vega/∂S

        Measures sensitivity of delta to changes in volatility.
        Critical for volatility trading and managing volatility exposure.

        Same for calls and puts.

        Formula: Vanna = -e^(-qT) * d2 * N'(d1) / σ

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Vanna value (per 1% vol change)
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Vanna = -e^(-qT) * d2 * N'(d1) / σ
        vanna = -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma

        # Scale for 1% vol change
        return vanna / 100

    @staticmethod
    def volga(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate Volga (Vomma) - second derivative of price w.r.t. volatility

        Volga = ∂²V/∂σ² = ∂Vega/∂σ

        Measures convexity of option price with respect to volatility.
        Important for managing vega risk and volatility smile dynamics.

        Same for calls and puts.

        Formula: Volga = Vega * d1 * d2 / σ

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Volga value (per 1% vol change squared)
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        vega_raw = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        # Volga = Vega * d1 * d2 / σ
        volga = vega_raw * d1 * d2 / sigma

        # Scale for 1% vol change (squared scaling)
        return volga / 10000

    @staticmethod
    def charm(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        option_type: str = "call"
    ) -> float:
        """
        Calculate Charm (Delta Decay/DdeltaDtime)

        Charm = ∂Δ/∂t = -∂Δ/∂T

        Measures how delta changes as time passes (delta decay).
        Essential for understanding how hedge ratios evolve.

        Formula (call):
            Charm = -q*e^(-qT)*N(d1) + e^(-qT)*N'(d1) * [2(r-q)T - d2*σ√T] / (2T*σ√T)

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            option_type: "call" or "put"

        Returns:
            Charm value (daily delta decay)
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        if option_type.lower() == "call":
            charm = (
                -q * np.exp(-q * T) * norm.cdf(d1)
                + np.exp(-q * T) * norm.pdf(d1) *
                (2 * (r - q) * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
            )
        else:
            charm = (
                q * np.exp(-q * T) * norm.cdf(-d1)
                + np.exp(-q * T) * norm.pdf(d1) *
                (2 * (r - q) * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
            )

        # Convert to daily (divide by 365)
        return charm / 365

    @staticmethod
    def veta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate Veta (DvegaDtime)

        Veta = ∂Vega/∂t

        Measures sensitivity of vega to passage of time.
        Important for managing long-dated options positions.

        Same for calls and puts.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Veta value (per day)
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Veta formula
        veta = (
            -S * np.exp(-q * T) * norm.pdf(d1) * sqrt_T *
            (q + (r - q) * d1 / (sigma * sqrt_T) - (1 + d1 * d2) / (2 * T))
        )

        # Convert to daily and scale for 1% vol
        return veta / 36500  # /365 for daily, /100 for 1% vol

    @staticmethod
    def speed(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate Speed (DgammaDspot or third derivative)

        Speed = ∂Γ/∂S = ∂³V/∂S³

        Measures rate of change of gamma with respect to spot.
        Important for large delta-hedged positions.

        Same for calls and puts.

        Formula: Speed = -Γ * (d1/(σ√T) + 1) / S

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Speed value
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)

        gamma = OptionsGreeks.gamma(S, K, T, r, sigma, q)

        # Speed = -Γ * (d1/(σ√T) + 1) / S
        speed = -gamma * (d1 / (sigma * sqrt_T) + 1) / S

        return speed

    @staticmethod
    def zomma(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate Zomma (DgammaDvol)

        Zomma = ∂Γ/∂σ

        Measures sensitivity of gamma to changes in volatility.
        Important for understanding gamma risk in different volatility regimes.

        Same for calls and puts.

        Formula: Zomma = Γ * (d1*d2 - 1) / σ

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Zomma value (per 1% vol change)
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        gamma = OptionsGreeks.gamma(S, K, T, r, sigma, q)

        # Zomma = Γ * (d1*d2 - 1) / σ
        zomma = gamma * (d1 * d2 - 1) / sigma

        # Scale for 1% vol change
        return zomma / 100

    @staticmethod
    def color(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate Color (DgammaDtime or Gamma Decay)

        Color = ∂Γ/∂t

        Measures rate of change of gamma with respect to time.
        Helps understand how gamma evolves as expiration approaches.

        Same for calls and puts.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Color value (daily)
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Color formula
        color = (
            -np.exp(-q * T) * norm.pdf(d1) / (2 * S * T * sigma * sqrt_T) *
            (2 * q * T + 1 + d1 * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (sigma * sqrt_T))
        )

        # Convert to daily
        return color / 365

    @staticmethod
    def ultima(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> float:
        """
        Calculate Ultima (DvolgaDvol or third derivative w.r.t. volatility)

        Ultima = ∂³V/∂σ³ = ∂Volga/∂σ

        Measures sensitivity of volga to changes in volatility.
        Used in advanced volatility trading strategies.

        Same for calls and puts.

        Formula: Ultima = -Vega * (d1*d2*(1-d1*d2) + d1² + d2²) / σ²

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield

        Returns:
            Ultima value
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        vega_raw = S * np.exp(-q * T) * norm.pdf(d1) * sqrt_T

        # Ultima = -Vega * (d1*d2*(1-d1*d2) + d1² + d2²) / σ²
        d1d2 = d1 * d2
        ultima = -vega_raw * (d1d2 * (1 - d1d2) + d1**2 + d2**2) / (sigma**2)

        # Scale for 1% vol change (cubed)
        return ultima / 1000000

    @staticmethod
    def calculate_all_greeks_extended(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0
    ) -> GreeksResult:
        """
        Calculate ALL Greeks (first and second order) for an option

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: "call" or "put"
            q: Dividend yield

        Returns:
            GreeksResult with all Greeks
        """
        # First-order Greeks
        if option_type.lower() == "call":
            price = OptionsGreeks.black_scholes_call(S, K, T, r, sigma, q)
            delta = OptionsGreeks.delta_call(S, K, T, r, sigma, q)
            theta = OptionsGreeks.theta_call(S, K, T, r, sigma, q)
            rho = OptionsGreeks.rho_call(S, K, T, r, sigma, q)
            intrinsic = max(S - K, 0)
        else:
            price = OptionsGreeks.black_scholes_put(S, K, T, r, sigma, q)
            delta = OptionsGreeks.delta_put(S, K, T, r, sigma, q)
            theta = OptionsGreeks.theta_put(S, K, T, r, sigma, q)
            rho = OptionsGreeks.rho_put(S, K, T, r, sigma, q)
            intrinsic = max(K - S, 0)

        gamma = OptionsGreeks.gamma(S, K, T, r, sigma, q)
        vega_val = OptionsGreeks.vega(S, K, T, r, sigma, q)

        # Second-order Greeks
        vanna = OptionsGreeks.vanna(S, K, T, r, sigma, q)
        volga = OptionsGreeks.volga(S, K, T, r, sigma, q)
        charm_val = OptionsGreeks.charm(S, K, T, r, sigma, q, option_type)
        veta = OptionsGreeks.veta(S, K, T, r, sigma, q)
        speed = OptionsGreeks.speed(S, K, T, r, sigma, q)
        zomma = OptionsGreeks.zomma(S, K, T, r, sigma, q)
        color = OptionsGreeks.color(S, K, T, r, sigma, q)
        ultima = OptionsGreeks.ultima(S, K, T, r, sigma, q)

        return GreeksResult(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega_val,
            rho=rho,
            vanna=vanna,
            volga=volga,
            charm=charm_val,
            veta=veta,
            speed=speed,
            zomma=zomma,
            color=color,
            ultima=ultima,
            price=price,
            intrinsic_value=intrinsic,
            extrinsic_value=price - intrinsic
        )

    @staticmethod
    def implied_volatility(
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        q: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method

        Args:
            option_price: Market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            option_type: "call" or "put"
            q: Dividend yield
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance

        Returns:
            Implied volatility (annual) or None if not converged
        """
        if T <= 0:
            return None

        # Initial guess
        sigma = 0.5

        for i in range(max_iterations):
            # Calculate option price with current sigma
            if option_type.lower() == "call":
                price = OptionsGreeks.black_scholes_call(S, K, T, r, sigma, q)
            else:
                price = OptionsGreeks.black_scholes_put(S, K, T, r, sigma, q)

            # Calculate vega for Newton-Raphson
            vega_val = OptionsGreeks.vega(S, K, T, r, sigma, q) * 100  # Multiply by 100 since we divided earlier

            # Check for convergence
            diff = option_price - price

            if abs(diff) < tolerance:
                return sigma

            if vega_val == 0:
                return None

            # Update sigma
            sigma = sigma + diff / vega_val

            # Ensure sigma stays positive
            if sigma <= 0:
                sigma = 0.01

        # Did not converge
        return None

    @staticmethod
    def calculate_all_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free interest rate
            sigma: Volatility (annual)
            option_type: "call" or "put"
            q: Dividend yield

        Returns:
            Dictionary with all Greeks and option price
        """
        if option_type.lower() == "call":
            price = OptionsGreeks.black_scholes_call(S, K, T, r, sigma, q)
            delta = OptionsGreeks.delta_call(S, K, T, r, sigma, q)
            theta = OptionsGreeks.theta_call(S, K, T, r, sigma, q)
            rho = OptionsGreeks.rho_call(S, K, T, r, sigma, q)
        else:
            price = OptionsGreeks.black_scholes_put(S, K, T, r, sigma, q)
            delta = OptionsGreeks.delta_put(S, K, T, r, sigma, q)
            theta = OptionsGreeks.theta_put(S, K, T, r, sigma, q)
            rho = OptionsGreeks.rho_put(S, K, T, r, sigma, q)

        gamma = OptionsGreeks.gamma(S, K, T, r, sigma, q)
        vega_val = OptionsGreeks.vega(S, K, T, r, sigma, q)

        # Calculate intrinsic and extrinsic value
        if option_type.lower() == "call":
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)

        extrinsic = price - intrinsic

        # Calculate moneyness
        if S > K:
            moneyness = "ITM" if option_type.lower() == "call" else "OTM"
        elif S < K:
            moneyness = "OTM" if option_type.lower() == "call" else "ITM"
        else:
            moneyness = "ATM"

        # Probability of profit (with drift term for accuracy)
        # Using risk-neutral measure: d = (ln(K/S) - (r - q - 0.5*σ²)*T) / (σ*√T)
        if option_type.lower() == "call":
            breakeven = K + price
            d = (np.log(breakeven / S) - (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            prob_profit = 1 - norm.cdf(d)
        else:
            breakeven = K - price
            d = (np.log(breakeven / S) - (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            prob_profit = norm.cdf(d)

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega_val,
            "rho": rho,
            "intrinsic_value": intrinsic,
            "extrinsic_value": extrinsic,
            "moneyness": moneyness,
            "breakeven": breakeven if option_type.lower() == "call" else breakeven,
            "probability_of_profit": prob_profit * 100
        }

    @staticmethod
    def days_to_expiration_to_years(days: int) -> float:
        """Convert days to expiration to years"""
        return days / 365.0

    @staticmethod
    def breakeven_price_call(K: float, premium: float) -> float:
        """Calculate breakeven price for call option"""
        return K + premium

    @staticmethod
    def breakeven_price_put(K: float, premium: float) -> float:
        """Calculate breakeven price for put option"""
        return K - premium

    @staticmethod
    def max_profit_call(premium: float) -> str:
        """Calculate max profit for long call"""
        return "Unlimited"

    @staticmethod
    def max_profit_put(K: float, premium: float) -> float:
        """Calculate max profit for long put"""
        return K - premium

    @staticmethod
    def max_loss_long_option(premium: float) -> float:
        """Calculate max loss for long option (call or put)"""
        return premium


def calculate_historical_volatility(prices: np.ndarray, window: int = 30) -> float:
    """
    Calculate historical volatility from price series

    Args:
        prices: Array of prices
        window: Lookback window for volatility calculation

    Returns:
        Annualized historical volatility
    """
    if len(prices) < window:
        window = len(prices)

    returns = np.diff(np.log(prices[-window:]))
    volatility = np.std(returns) * np.sqrt(252)  # Annualize (252 trading days)

    return volatility


def calculate_option_profit_loss(
    entry_price: float,
    current_price: float,
    contracts: int,
    option_type: str = "long"
) -> Dict[str, float]:
    """
    Calculate P&L for options position

    Args:
        entry_price: Entry price per option
        current_price: Current price per option
        contracts: Number of contracts (1 contract = 100 shares)
        option_type: "long" or "short"

    Returns:
        Dictionary with P&L metrics
    """
    multiplier = 100  # Options multiplier

    if option_type.lower() == "long":
        pnl_per_contract = (current_price - entry_price) * multiplier
    else:  # short
        pnl_per_contract = (entry_price - current_price) * multiplier

    total_pnl = pnl_per_contract * contracts
    pnl_pct = ((current_price - entry_price) / entry_price) * 100

    if option_type.lower() == "short":
        pnl_pct = -pnl_pct

    return {
        "pnl_per_contract": pnl_per_contract,
        "total_pnl": total_pnl,
        "pnl_percentage": pnl_pct,
        "entry_price": entry_price,
        "current_price": current_price,
        "contracts": contracts
    }


# =============================================================================
# American Options Pricing - Binomial Tree (Cox-Ross-Rubinstein Model)
# =============================================================================

class BinomialTree:
    """
    Cox-Ross-Rubinstein (CRR) Binomial Tree for American Options Pricing

    The binomial model is essential for American options because:
    1. American options can be exercised at any time before expiration
    2. Black-Scholes assumes European exercise only
    3. Early exercise premium is significant for puts on non-dividend stocks
       and calls on dividend-paying stocks

    CRR Parameters:
    - u = exp(σ√Δt)  - up factor
    - d = 1/u = exp(-σ√Δt)  - down factor
    - p = (exp((r-q)Δt) - d) / (u - d)  - risk-neutral probability
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        steps: int = 100
    ):
        """
        Initialize binomial tree

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            q: Dividend yield
            steps: Number of time steps (more = more accurate, slower)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.steps = steps

        # Calculate CRR parameters
        self.dt = T / steps
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp((r - q) * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-r * self.dt)

    def price_american_call(self) -> Dict:
        """
        Price American call option using binomial tree

        Returns:
            Dictionary with price, greeks, and early exercise boundary
        """
        return self._price_option("call", american=True)

    def price_american_put(self) -> Dict:
        """
        Price American put option using binomial tree

        Returns:
            Dictionary with price, greeks, and early exercise boundary
        """
        return self._price_option("put", american=True)

    def price_european_call(self) -> Dict:
        """
        Price European call using binomial tree (for comparison)
        """
        return self._price_option("call", american=False)

    def price_european_put(self) -> Dict:
        """
        Price European put using binomial tree (for comparison)
        """
        return self._price_option("put", american=False)

    def _price_option(self, option_type: str, american: bool = True) -> Dict:
        """
        Core binomial tree pricing algorithm

        Args:
            option_type: "call" or "put"
            american: If True, check for early exercise at each node

        Returns:
            Dictionary with price, delta, gamma, early_exercise_boundary
        """
        n = self.steps

        # Build price tree (only need to store current and next layer)
        # Stock prices at maturity
        stock_prices = np.array([
            self.S * (self.u ** (n - i)) * (self.d ** i)
            for i in range(n + 1)
        ])

        # Option values at maturity
        if option_type.lower() == "call":
            option_values = np.maximum(stock_prices - self.K, 0)
        else:
            option_values = np.maximum(self.K - stock_prices, 0)

        # Track early exercise boundary (for puts especially)
        early_exercise_boundary = []

        # Work backwards through the tree
        for j in range(n - 1, -1, -1):
            # Stock prices at this time step
            stock_at_j = np.array([
                self.S * (self.u ** (j - i)) * (self.d ** i)
                for i in range(j + 1)
            ])

            # Expected option value (risk-neutral)
            expected_value = (
                self.p * option_values[:-1] +
                (1 - self.p) * option_values[1:]
            ) * self.discount

            # Intrinsic value
            if option_type.lower() == "call":
                intrinsic = np.maximum(stock_at_j - self.K, 0)
            else:
                intrinsic = np.maximum(self.K - stock_at_j, 0)

            if american:
                # American: take max of holding vs exercising
                option_values = np.maximum(expected_value, intrinsic)

                # Track early exercise boundary (where exercise is optimal)
                exercise_optimal = expected_value < intrinsic
                if np.any(exercise_optimal) and option_type.lower() == "put":
                    # Find critical stock price for early exercise
                    idx = np.where(exercise_optimal)[0]
                    if len(idx) > 0:
                        early_exercise_boundary.append({
                            'time': j * self.dt,
                            'critical_price': stock_at_j[idx[-1]]
                        })
            else:
                # European: just hold
                option_values = expected_value

        price = option_values[0]

        # Calculate delta and gamma using the tree
        # Delta: finite difference at first time step
        if n >= 1:
            f_u = self.S * self.u  # Price after up move
            f_d = self.S * self.d  # Price after down move

            # Need option values at t=dt
            stock_at_1 = np.array([self.S * self.u, self.S * self.d])
            if option_type.lower() == "call":
                intrinsic_1 = np.maximum(stock_at_1 - self.K, 0)
            else:
                intrinsic_1 = np.maximum(self.K - stock_at_1, 0)

            # Re-run to get values at t=dt
            _, delta, gamma = self._calculate_greeks(option_type, american)
        else:
            delta = 0.0
            gamma = 0.0

        # Early exercise premium
        if american:
            european_price = self._price_option(option_type, american=False)['price']
            early_exercise_premium = price - european_price
        else:
            early_exercise_premium = 0.0

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'early_exercise_premium': early_exercise_premium,
            'early_exercise_boundary': early_exercise_boundary,
            'is_american': american,
            'steps': self.steps
        }

    def _calculate_greeks(self, option_type: str, american: bool) -> Tuple[float, float, float]:
        """
        Calculate delta and gamma from the tree using finite differences

        Uses a simpler, non-recursive approach to avoid infinite loops.
        """
        eps = 0.01 * self.S

        # Price at S using the core algorithm directly (without calling Greeks again)
        v0 = self._price_core(option_type, american)

        # Price at S+eps
        orig_S = self.S
        self.S = orig_S + eps
        v_up = self._price_core(option_type, american)

        # Price at S-eps
        self.S = orig_S - eps
        v_down = self._price_core(option_type, american)

        # Restore original S
        self.S = orig_S

        delta = (v_up - v_down) / (2 * eps)
        gamma = (v_up - 2 * v0 + v_down) / (eps ** 2)

        return v0, delta, gamma

    def _price_core(self, option_type: str, american: bool) -> float:
        """
        Core pricing without Greek calculation (avoids recursion)
        """
        n = self.steps

        # Stock prices at maturity
        stock_prices = np.array([
            self.S * (self.u ** (n - i)) * (self.d ** i)
            for i in range(n + 1)
        ])

        # Option values at maturity
        if option_type.lower() == "call":
            option_values = np.maximum(stock_prices - self.K, 0)
        else:
            option_values = np.maximum(self.K - stock_prices, 0)

        # Work backwards through the tree
        for j in range(n - 1, -1, -1):
            stock_at_j = np.array([
                self.S * (self.u ** (j - i)) * (self.d ** i)
                for i in range(j + 1)
            ])

            expected_value = (
                self.p * option_values[:-1] +
                (1 - self.p) * option_values[1:]
            ) * self.discount

            if option_type.lower() == "call":
                intrinsic = np.maximum(stock_at_j - self.K, 0)
            else:
                intrinsic = np.maximum(self.K - stock_at_j, 0)

            if american:
                option_values = np.maximum(expected_value, intrinsic)
            else:
                option_values = expected_value

        return option_values[0]


def price_american_option(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "put",
    q: float = 0.0,
    steps: int = 100
) -> Dict:
    """
    Convenience function to price American options

    Args:
        S: Stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility
        option_type: "call" or "put"
        q: Dividend yield
        steps: Number of tree steps

    Returns:
        Dictionary with pricing results

    Example:
        >>> result = price_american_option(100, 100, 0.5, 0.05, 0.2, "put")
        >>> print(f"American Put: ${result['price']:.2f}")
        >>> print(f"Early Exercise Premium: ${result['early_exercise_premium']:.4f}")
    """
    tree = BinomialTree(S, K, T, r, sigma, q, steps)

    if option_type.lower() == "call":
        return tree.price_american_call()
    else:
        return tree.price_american_put()


# =============================================================================
# Implied Volatility Surface Construction (SVI Model)
# =============================================================================

class IVSurfaceBuilder:
    """
    Build Implied Volatility Surface using SVI (Stochastic Volatility Inspired)
    parameterization

    The SVI model is industry-standard for volatility surface fitting because:
    1. Guarantees no calendar arbitrage with proper constraints
    2. Captures both smile and skew with intuitive parameters
    3. Efficient to fit (5 parameters per expiry)

    SVI Raw Parameterization:
        w(k) = a + b*(ρ*(k-m) + sqrt((k-m)² + σ²))

    where:
        k = log(K/F) is log-moneyness (F = forward price)
        w = σ² * T is total variance
    """

    def __init__(
        self,
        spot: float,
        rate: float = 0.05,
        dividend_yield: float = 0.0
    ):
        """
        Initialize IV surface builder

        Args:
            spot: Current spot price
            rate: Risk-free rate
            dividend_yield: Continuous dividend yield
        """
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield

    def fit_svi_slice(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        expiry: float,
        weights: Optional[np.ndarray] = None
    ) -> SVIParams:
        """
        Fit SVI parameters to a single expiry slice

        Args:
            strikes: Array of strike prices
            ivs: Array of implied volatilities
            expiry: Time to expiration (years)
            weights: Optional weights for fitting (e.g., inverse bid-ask spread)

        Returns:
            SVIParams for this expiry
        """
        forward = self.spot * np.exp((self.rate - self.dividend_yield) * expiry)
        log_moneyness = np.log(strikes / forward)
        total_variance = ivs ** 2 * expiry

        if weights is None:
            weights = np.ones_like(strikes)

        def svi_variance(params, k):
            a, b, rho, m, sigma = params
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

        def objective(params):
            a, b, rho, m, sigma = params
            model_var = svi_variance(params, log_moneyness)
            return np.sum(weights * (model_var - total_variance)**2)

        # Initial guess
        atm_var = np.interp(0, log_moneyness, total_variance)
        x0 = [atm_var, 0.1, -0.3, 0.0, 0.1]

        # Bounds to ensure no arbitrage
        bounds = [
            (0, None),      # a >= 0
            (0, None),      # b >= 0
            (-0.999, 0.999),# -1 < ρ < 1
            (-0.5, 0.5),    # m centered around 0
            (0.001, 1.0)    # σ > 0
        ]

        # Constraints: a + b*σ*sqrt(1-ρ²) >= 0 (butterfly arbitrage)
        def butterfly_constraint(params):
            a, b, rho, m, sigma = params
            return a + b * sigma * np.sqrt(1 - rho**2)

        constraints = {'type': 'ineq', 'fun': butterfly_constraint}

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        a, b, rho, m, sigma = result.x
        return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)

    def build_surface(
        self,
        market_data: List[Dict],
        fit_method: str = 'svi'
    ) -> IVSurface:
        """
        Build complete IV surface from market data

        Args:
            market_data: List of dicts with keys:
                - 'strike': strike price
                - 'expiry': time to expiration (years)
                - 'iv': implied volatility
                - 'bid': optional bid price
                - 'ask': optional ask price
                - 'price': market price
            fit_method: 'svi' or 'spline'

        Returns:
            IVSurface object

        Example:
            >>> builder = IVSurfaceBuilder(spot=100, rate=0.05)
            >>> data = [
            ...     {'strike': 95, 'expiry': 0.25, 'iv': 0.22, 'price': 6.5},
            ...     {'strike': 100, 'expiry': 0.25, 'iv': 0.20, 'price': 3.2},
            ...     {'strike': 105, 'expiry': 0.25, 'iv': 0.21, 'price': 1.1},
            ...     # ... more data points
            ... ]
            >>> surface = builder.build_surface(data)
            >>> iv = surface.get_iv(strike=102, expiry=0.3)
        """
        surface = IVSurface(
            spot=self.spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield
        )

        # Convert to points
        for item in market_data:
            point = IVSurfacePoint(
                strike=item['strike'],
                expiry=item['expiry'],
                implied_vol=item['iv'],
                market_price=item.get('price', 0),
                bid=item.get('bid'),
                ask=item.get('ask')
            )
            surface.points.append(point)

        # Group by expiry and fit SVI for each
        expiries = sorted(set(p.expiry for p in surface.points))

        for expiry in expiries:
            expiry_points = [p for p in surface.points if p.expiry == expiry]

            if len(expiry_points) < 3:
                # Need at least 3 points to fit SVI
                continue

            strikes = np.array([p.strike for p in expiry_points])
            ivs = np.array([p.implied_vol for p in expiry_points])

            # Use bid-ask spread as weights if available
            weights = None
            if all(p.bid is not None and p.ask is not None for p in expiry_points):
                spreads = np.array([p.ask - p.bid for p in expiry_points])
                weights = 1 / (spreads + 0.01)  # Inverse spread weighting

            svi_params = self.fit_svi_slice(strikes, ivs, expiry, weights)
            surface.svi_params[expiry] = svi_params

        return surface

    def calculate_local_vol(
        self,
        surface: IVSurface,
        strike: float,
        expiry: float
    ) -> float:
        """
        Calculate Dupire local volatility from IV surface

        Local vol = σ(K,T) for the local volatility model dS = μS dt + σ(S,t)S dW

        Dupire formula:
            σ_local² = (∂w/∂T) / (1 - (k/w)*(∂w/∂k) + 0.25*(-0.25 - 1/w + k²/w²)*(∂w/∂k)² + 0.5*(∂²w/∂k²))

        Args:
            surface: IVSurface object
            strike: Strike price
            expiry: Time to expiration

        Returns:
            Local volatility at (K, T)
        """
        forward = self.spot * np.exp((self.rate - self.dividend_yield) * expiry)
        k = np.log(strike / forward)

        iv = surface.get_iv(strike, expiry)
        if iv is None:
            return 0.0

        w = iv ** 2 * expiry  # Total variance

        # Numerical derivatives
        dk = 0.001
        dt = 0.001

        iv_k_up = surface.get_iv(strike * np.exp(dk), expiry)
        iv_k_down = surface.get_iv(strike * np.exp(-dk), expiry)

        if iv_k_up is None or iv_k_down is None:
            return iv  # Fall back to Black vol

        w_k_up = (iv_k_up ** 2) * expiry
        w_k_down = (iv_k_down ** 2) * expiry

        dw_dk = (w_k_up - w_k_down) / (2 * dk)
        d2w_dk2 = (w_k_up - 2 * w + w_k_down) / (dk ** 2)

        # Time derivative
        if expiry + dt <= max(surface.svi_params.keys()):
            iv_t_up = surface.get_iv(strike, expiry + dt)
            if iv_t_up is not None:
                w_t_up = (iv_t_up ** 2) * (expiry + dt)
                dw_dt = (w_t_up - w) / dt
            else:
                dw_dt = iv ** 2  # Assume constant variance rate
        else:
            dw_dt = iv ** 2

        # Dupire formula
        numerator = dw_dt
        denominator = (
            1 - (k / w) * dw_dk
            + 0.25 * (-0.25 - 1/w + k**2/w**2) * dw_dk**2
            + 0.5 * d2w_dk2
        )

        if denominator <= 0:
            return iv  # Fallback

        local_var = numerator / denominator
        return np.sqrt(max(local_var, 0))


# =============================================================================
# Options Risk Dashboard
# =============================================================================

class OptionsRiskDashboard:
    """
    Comprehensive options portfolio risk analysis

    Aggregates Greeks across positions, runs stress tests,
    and provides risk metrics for options portfolios.
    """

    def __init__(self, spot: float, rate: float = 0.05, dividend_yield: float = 0.0):
        """
        Initialize risk dashboard

        Args:
            spot: Current underlying price
            rate: Risk-free rate
            dividend_yield: Dividend yield
        """
        self.spot = spot
        self.rate = rate
        self.dividend_yield = dividend_yield
        self.positions: List[Dict] = []

    def add_position(
        self,
        strike: float,
        expiry: float,
        option_type: str,
        quantity: int,
        sigma: float,
        entry_price: Optional[float] = None
    ):
        """
        Add an options position to the portfolio

        Args:
            strike: Strike price
            expiry: Time to expiration (years)
            option_type: "call" or "put"
            quantity: Number of contracts (positive for long, negative for short)
            sigma: Implied volatility
            entry_price: Entry price per option (for P&L tracking)
        """
        position = {
            'strike': strike,
            'expiry': expiry,
            'option_type': option_type,
            'quantity': quantity,
            'sigma': sigma,
            'entry_price': entry_price
        }
        self.positions.append(position)

    def clear_positions(self):
        """Clear all positions"""
        self.positions = []

    def calculate_portfolio_greeks(self) -> OptionsRiskReport:
        """
        Calculate aggregate portfolio Greeks

        Returns:
            OptionsRiskReport with complete risk metrics
        """
        report = OptionsRiskReport()

        multiplier = 100  # Options contract multiplier

        for pos in self.positions:
            greeks = OptionsGreeks.calculate_all_greeks_extended(
                S=self.spot,
                K=pos['strike'],
                T=pos['expiry'],
                r=self.rate,
                sigma=pos['sigma'],
                option_type=pos['option_type'],
                q=self.dividend_yield
            )

            qty = pos['quantity']

            # Aggregate first-order Greeks
            report.total_delta += greeks.delta * qty * multiplier
            report.total_gamma += greeks.gamma * qty * multiplier
            report.total_vega += greeks.vega * qty * multiplier
            report.total_theta += greeks.theta * qty * multiplier

            # Aggregate second-order Greeks
            report.total_vanna += greeks.vanna * qty * multiplier
            report.total_volga += greeks.volga * qty * multiplier
            report.total_charm += greeks.charm * qty * multiplier

            # Store position detail
            pos_detail = {
                'strike': pos['strike'],
                'expiry': pos['expiry'],
                'type': pos['option_type'],
                'quantity': qty,
                'price': greeks.price,
                'delta': greeks.delta * qty * multiplier,
                'gamma': greeks.gamma * qty * multiplier,
                'vega': greeks.vega * qty * multiplier,
                'theta': greeks.theta * qty * multiplier,
                'vanna': greeks.vanna * qty * multiplier,
                'volga': greeks.volga * qty * multiplier
            }
            report.positions.append(pos_detail)

        # Dollar-denominated risks
        report.delta_dollars = report.total_delta  # $ per $1 move
        report.gamma_dollars = report.total_gamma * self.spot / 100  # $ per 1% move
        report.vega_dollars = report.total_vega  # $ per 1% IV move
        report.theta_dollars = report.total_theta  # $ per day

        # Scenario analysis
        report.pnl_up_1pct = self._scenario_pnl(spot_change=0.01, vol_change=0)
        report.pnl_down_1pct = self._scenario_pnl(spot_change=-0.01, vol_change=0)
        report.pnl_vol_up = self._scenario_pnl(spot_change=0, vol_change=0.01)
        report.pnl_vol_down = self._scenario_pnl(spot_change=0, vol_change=-0.01)
        report.pnl_1day = report.theta_dollars

        # Estimate max loss (simplified)
        report.max_loss_estimate = self._estimate_max_loss()

        # 95% VaR (parametric, assuming normal returns)
        report.var_95 = self._calculate_var(0.95, total_delta=report.total_delta)

        return report

    def _scenario_pnl(self, spot_change: float, vol_change: float) -> float:
        """
        Calculate P&L for a given scenario

        Args:
            spot_change: Percentage change in spot (e.g., 0.01 for +1%)
            vol_change: Absolute change in vol (e.g., 0.01 for +1%)

        Returns:
            Expected P&L
        """
        new_spot = self.spot * (1 + spot_change)
        multiplier = 100
        pnl = 0.0

        for pos in self.positions:
            new_sigma = pos['sigma'] + vol_change

            # Current value
            current_price = (
                OptionsGreeks.black_scholes_call(
                    self.spot, pos['strike'], pos['expiry'],
                    self.rate, pos['sigma'], self.dividend_yield
                ) if pos['option_type'].lower() == 'call' else
                OptionsGreeks.black_scholes_put(
                    self.spot, pos['strike'], pos['expiry'],
                    self.rate, pos['sigma'], self.dividend_yield
                )
            )

            # New value
            new_price = (
                OptionsGreeks.black_scholes_call(
                    new_spot, pos['strike'], pos['expiry'],
                    self.rate, new_sigma, self.dividend_yield
                ) if pos['option_type'].lower() == 'call' else
                OptionsGreeks.black_scholes_put(
                    new_spot, pos['strike'], pos['expiry'],
                    self.rate, new_sigma, self.dividend_yield
                )
            )

            pnl += (new_price - current_price) * pos['quantity'] * multiplier

        return pnl

    def _estimate_max_loss(self) -> float:
        """
        Estimate maximum loss for the portfolio

        For long options: max loss is premium paid
        For short options: can be unlimited (calls) or strike minus premium (puts)
        """
        max_loss = 0.0
        multiplier = 100

        for pos in self.positions:
            price = (
                OptionsGreeks.black_scholes_call(
                    self.spot, pos['strike'], pos['expiry'],
                    self.rate, pos['sigma'], self.dividend_yield
                ) if pos['option_type'].lower() == 'call' else
                OptionsGreeks.black_scholes_put(
                    self.spot, pos['strike'], pos['expiry'],
                    self.rate, pos['sigma'], self.dividend_yield
                )
            )

            if pos['quantity'] > 0:  # Long position
                max_loss += price * pos['quantity'] * multiplier
            else:  # Short position
                if pos['option_type'].lower() == 'call':
                    # Short calls have unlimited theoretical loss
                    # Estimate as 3x current price move
                    max_loss += (3 * self.spot - pos['strike'] + price) * abs(pos['quantity']) * multiplier
                else:
                    # Short puts max loss is strike - premium
                    max_loss += (pos['strike'] - price) * abs(pos['quantity']) * multiplier

        return max_loss

    def _calculate_var(self, confidence: float = 0.95, total_delta: Optional[float] = None) -> float:
        """
        Calculate parametric VaR

        Uses delta-normal approximation

        Args:
            confidence: Confidence level (default 0.95)
            total_delta: Pre-calculated portfolio delta (to avoid recursion)
        """
        # Assume 20% annual vol for underlying
        underlying_vol = 0.20
        daily_vol = underlying_vol / np.sqrt(252)

        # Delta-normal VaR
        z_score = norm.ppf(confidence)

        if total_delta is None:
            # Calculate delta directly to avoid recursion
            total_delta = 0.0
            multiplier = 100
            for pos in self.positions:
                if pos['option_type'].lower() == 'call':
                    delta = OptionsGreeks.delta_call(
                        self.spot, pos['strike'], pos['expiry'],
                        self.rate, pos['sigma'], self.dividend_yield
                    )
                else:
                    delta = OptionsGreeks.delta_put(
                        self.spot, pos['strike'], pos['expiry'],
                        self.rate, pos['sigma'], self.dividend_yield
                    )
                total_delta += delta * pos['quantity'] * multiplier

        portfolio_var = abs(total_delta) * self.spot * daily_vol * z_score

        return portfolio_var

    def stress_test(
        self,
        scenarios: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Run stress tests on the portfolio

        Args:
            scenarios: List of scenarios, each with:
                - 'name': scenario name
                - 'spot_change': % change in spot
                - 'vol_change': absolute change in IV

        Returns:
            List of scenario results
        """
        if scenarios is None:
            scenarios = [
                {'name': 'Market Crash (-20%)', 'spot_change': -0.20, 'vol_change': 0.20},
                {'name': 'Correction (-10%)', 'spot_change': -0.10, 'vol_change': 0.10},
                {'name': 'Rally (+10%)', 'spot_change': 0.10, 'vol_change': -0.05},
                {'name': 'Vol Spike (flat)', 'spot_change': 0, 'vol_change': 0.15},
                {'name': 'Vol Crush', 'spot_change': 0, 'vol_change': -0.10},
                {'name': 'Black Swan (-30%, +40% vol)', 'spot_change': -0.30, 'vol_change': 0.40},
            ]

        results = []
        for scenario in scenarios:
            pnl = self._scenario_pnl(
                spot_change=scenario['spot_change'],
                vol_change=scenario['vol_change']
            )
            results.append({
                'scenario': scenario['name'],
                'spot_change': scenario['spot_change'],
                'vol_change': scenario['vol_change'],
                'pnl': pnl,
                'new_spot': self.spot * (1 + scenario['spot_change'])
            })

        return results

    def generate_report(self) -> str:
        """
        Generate formatted risk report

        Returns:
            Formatted string report
        """
        greeks = self.calculate_portfolio_greeks()
        stress = self.stress_test()

        report = []
        report.append("=" * 60)
        report.append("OPTIONS PORTFOLIO RISK REPORT")
        report.append("=" * 60)
        report.append(f"\nUnderlying: ${self.spot:.2f}")
        report.append(f"Positions: {len(self.positions)}")

        report.append("\n--- AGGREGATE GREEKS ---")
        report.append(f"Delta:  {greeks.total_delta:>10.2f}  (${greeks.delta_dollars:>10.2f}/1pt)")
        report.append(f"Gamma:  {greeks.total_gamma:>10.4f}  (${greeks.gamma_dollars:>10.2f}/1%)")
        report.append(f"Vega:   {greeks.total_vega:>10.2f}  (${greeks.vega_dollars:>10.2f}/1%)")
        report.append(f"Theta:  {greeks.total_theta:>10.2f}  (${greeks.theta_dollars:>10.2f}/day)")

        report.append("\n--- SECOND-ORDER GREEKS ---")
        report.append(f"Vanna:  {greeks.total_vanna:>10.4f}")
        report.append(f"Volga:  {greeks.total_volga:>10.4f}")
        report.append(f"Charm:  {greeks.total_charm:>10.4f}")

        report.append("\n--- SCENARIO ANALYSIS ---")
        report.append(f"Spot +1%:   ${greeks.pnl_up_1pct:>10.2f}")
        report.append(f"Spot -1%:   ${greeks.pnl_down_1pct:>10.2f}")
        report.append(f"Vol +1%:    ${greeks.pnl_vol_up:>10.2f}")
        report.append(f"Vol -1%:    ${greeks.pnl_vol_down:>10.2f}")
        report.append(f"1-Day Decay: ${greeks.pnl_1day:>10.2f}")

        report.append("\n--- RISK METRICS ---")
        report.append(f"Max Loss Estimate:  ${greeks.max_loss_estimate:>10.2f}")
        report.append(f"95% VaR (1-day):    ${greeks.var_95:>10.2f}")

        report.append("\n--- STRESS TESTS ---")
        for s in stress:
            report.append(f"{s['scenario']:<30} P&L: ${s['pnl']:>10.2f}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# =============================================================================
# Monte Carlo Options Pricing
# =============================================================================

@dataclass
class MonteCarloResult:
    """Results from Monte Carlo options pricing"""
    price: float
    standard_error: float
    confidence_interval_95: Tuple[float, float]
    n_simulations: int
    n_timesteps: int
    convergence_achieved: bool

    # Greeks estimated via finite difference
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None

    # Additional metrics
    exercise_probability: Optional[float] = None  # For American options
    early_exercise_premium: Optional[float] = None  # American - European difference
    path_prices: Optional[np.ndarray] = None  # Sample of terminal prices


class MonteCarloOptionsPricer:
    """
    Monte Carlo simulation for exotic options pricing.

    Supports:
    - European options (vanilla)
    - Asian options (arithmetic & geometric average)
    - Barrier options (knock-in/knock-out, up/down)
    - Lookback options (fixed/floating strike)
    - American options (Longstaff-Schwartz LSM method)

    ★ Insight ─────────────────────────────────────
    Monte Carlo is essential for path-dependent options where
    Black-Scholes has no closed-form solution. The method simulates
    thousands of price paths under the risk-neutral measure and
    averages discounted payoffs.
    ─────────────────────────────────────────────────

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate (annualized)
        sigma: Volatility (annualized)
        q: Dividend yield (annualized)

    Example:
        >>> pricer = MonteCarloOptionsPricer(S=100, K=100, T=1, r=0.05, sigma=0.2)
        >>> result = pricer.price_european_call(n_simulations=100000)
        >>> print(f"Price: ${result.price:.4f} ± ${result.standard_error:.4f}")
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        seed: Optional[int] = None
    ):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

        if seed is not None:
            np.random.seed(seed)

    def _generate_paths(
        self,
        n_simulations: int,
        n_timesteps: int,
        antithetic: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate GBM price paths using the risk-neutral measure.

        Args:
            n_simulations: Number of simulation paths
            n_timesteps: Number of time steps
            antithetic: Use antithetic variates for variance reduction

        Returns:
            (paths, times) - paths shape: (n_simulations, n_timesteps+1)
        """
        dt = self.T / n_timesteps
        times = np.linspace(0, self.T, n_timesteps + 1)

        # Generate random numbers
        if antithetic:
            n_half = n_simulations // 2
            Z = np.random.standard_normal((n_half, n_timesteps))
            Z = np.vstack([Z, -Z])  # Antithetic pairs
        else:
            Z = np.random.standard_normal((n_simulations, n_timesteps))

        # Drift and diffusion under risk-neutral measure
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        # Generate log returns
        log_returns = drift + diffusion * Z

        # Build paths
        paths = np.zeros((n_simulations, n_timesteps + 1))
        paths[:, 0] = self.S
        paths[:, 1:] = self.S * np.exp(np.cumsum(log_returns, axis=1))

        return paths, times

    def _calculate_greeks(
        self,
        base_price: float,
        n_simulations: int,
        n_timesteps: int,
        pricer_func,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate Greeks via finite difference method.

        Args:
            base_price: Base option price
            n_simulations: Number of simulations
            n_timesteps: Number of timesteps
            pricer_func: Function to call for pricing
            **kwargs: Additional arguments for pricer function

        Returns:
            Dictionary with delta, gamma, vega, theta
        """
        bump_s = 0.01 * self.S
        bump_v = 0.01
        bump_t = 1 / 365

        # Delta and Gamma
        pricer_up = MonteCarloOptionsPricer(
            self.S + bump_s, self.K, self.T, self.r, self.sigma, self.q
        )
        pricer_down = MonteCarloOptionsPricer(
            self.S - bump_s, self.K, self.T, self.r, self.sigma, self.q
        )

        price_up = getattr(pricer_up, pricer_func)(n_simulations, n_timesteps, **kwargs).price
        price_down = getattr(pricer_down, pricer_func)(n_simulations, n_timesteps, **kwargs).price

        delta = (price_up - price_down) / (2 * bump_s)
        gamma = (price_up - 2 * base_price + price_down) / (bump_s**2)

        # Vega
        pricer_vega_up = MonteCarloOptionsPricer(
            self.S, self.K, self.T, self.r, self.sigma + bump_v, self.q
        )
        price_vega_up = getattr(pricer_vega_up, pricer_func)(n_simulations, n_timesteps, **kwargs).price
        vega = (price_vega_up - base_price) / bump_v

        # Theta
        if self.T > bump_t:
            pricer_theta = MonteCarloOptionsPricer(
                self.S, self.K, self.T - bump_t, self.r, self.sigma, self.q
            )
            price_theta = getattr(pricer_theta, pricer_func)(n_simulations, n_timesteps, **kwargs).price
            theta = (price_theta - base_price) / bump_t
        else:
            theta = 0.0

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }

    def price_european_call(
        self,
        n_simulations: int = 100000,
        n_timesteps: int = 252,
        calculate_greeks: bool = False
    ) -> MonteCarloResult:
        """
        Price European call option using Monte Carlo.

        For validation, compare with Black-Scholes closed-form solution.
        """
        paths, _ = self._generate_paths(n_simulations, n_timesteps)
        terminal_prices = paths[:, -1]

        # European call payoff
        payoffs = np.maximum(terminal_prices - self.K, 0)
        discount = np.exp(-self.r * self.T)
        discounted_payoffs = discount * payoffs

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)

        result = MonteCarloResult(
            price=price,
            standard_error=std_error,
            confidence_interval_95=ci_95,
            n_simulations=n_simulations,
            n_timesteps=n_timesteps,
            convergence_achieved=std_error < 0.01 * price if price > 0 else True,
            path_prices=terminal_prices[:100]
        )

        if calculate_greeks:
            greeks = self._calculate_greeks(
                price, n_simulations // 10, n_timesteps,
                'price_european_call', calculate_greeks=False
            )
            result.delta = greeks['delta']
            result.gamma = greeks['gamma']
            result.vega = greeks['vega']
            result.theta = greeks['theta']

        return result

    def price_european_put(
        self,
        n_simulations: int = 100000,
        n_timesteps: int = 252,
        calculate_greeks: bool = False
    ) -> MonteCarloResult:
        """Price European put option using Monte Carlo."""
        paths, _ = self._generate_paths(n_simulations, n_timesteps)
        terminal_prices = paths[:, -1]

        # European put payoff
        payoffs = np.maximum(self.K - terminal_prices, 0)
        discount = np.exp(-self.r * self.T)
        discounted_payoffs = discount * payoffs

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)

        result = MonteCarloResult(
            price=price,
            standard_error=std_error,
            confidence_interval_95=ci_95,
            n_simulations=n_simulations,
            n_timesteps=n_timesteps,
            convergence_achieved=std_error < 0.01 * price if price > 0 else True,
            path_prices=terminal_prices[:100]
        )

        if calculate_greeks:
            greeks = self._calculate_greeks(
                price, n_simulations // 10, n_timesteps,
                'price_european_put', calculate_greeks=False
            )
            result.delta = greeks['delta']
            result.gamma = greeks['gamma']
            result.vega = greeks['vega']
            result.theta = greeks['theta']

        return result

    def price_asian_call(
        self,
        n_simulations: int = 100000,
        n_timesteps: int = 252,
        average_type: str = 'arithmetic',
        calculate_greeks: bool = False
    ) -> MonteCarloResult:
        """
        Price Asian call option (average price option).

        Asian options have payoff based on average price over the life,
        making them less volatile and cheaper than vanilla options.

        Args:
            n_simulations: Number of simulation paths
            n_timesteps: Number of averaging points
            average_type: 'arithmetic' or 'geometric'
            calculate_greeks: Whether to estimate Greeks

        Returns:
            MonteCarloResult with price and statistics
        """
        paths, _ = self._generate_paths(n_simulations, n_timesteps)

        # Calculate average price along each path
        if average_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        elif average_type == 'geometric':
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))
        else:
            raise ValueError(f"Unknown average_type: {average_type}")

        # Asian call payoff
        payoffs = np.maximum(avg_prices - self.K, 0)
        discount = np.exp(-self.r * self.T)
        discounted_payoffs = discount * payoffs

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)

        result = MonteCarloResult(
            price=price,
            standard_error=std_error,
            confidence_interval_95=ci_95,
            n_simulations=n_simulations,
            n_timesteps=n_timesteps,
            convergence_achieved=std_error < 0.01 * price if price > 0 else True
        )

        if calculate_greeks:
            greeks = self._calculate_greeks(
                price, n_simulations // 10, n_timesteps,
                'price_asian_call', calculate_greeks=False, average_type=average_type
            )
            result.delta = greeks['delta']
            result.gamma = greeks['gamma']
            result.vega = greeks['vega']
            result.theta = greeks['theta']

        return result

    def price_asian_put(
        self,
        n_simulations: int = 100000,
        n_timesteps: int = 252,
        average_type: str = 'arithmetic',
        calculate_greeks: bool = False
    ) -> MonteCarloResult:
        """Price Asian put option (average price option)."""
        paths, _ = self._generate_paths(n_simulations, n_timesteps)

        if average_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        elif average_type == 'geometric':
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))
        else:
            raise ValueError(f"Unknown average_type: {average_type}")

        payoffs = np.maximum(self.K - avg_prices, 0)
        discount = np.exp(-self.r * self.T)
        discounted_payoffs = discount * payoffs

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)

        return MonteCarloResult(
            price=price,
            standard_error=std_error,
            confidence_interval_95=ci_95,
            n_simulations=n_simulations,
            n_timesteps=n_timesteps,
            convergence_achieved=std_error < 0.01 * price if price > 0 else True
        )

    def price_barrier_option(
        self,
        barrier: float,
        barrier_type: str,  # 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
        option_type: str = 'call',  # 'call' or 'put'
        n_simulations: int = 100000,
        n_timesteps: int = 252,
        rebate: float = 0.0
    ) -> MonteCarloResult:
        """
        Price barrier option using Monte Carlo.

        Barrier options activate (knock-in) or deactivate (knock-out)
        when the underlying crosses a specified barrier level.

        Args:
            barrier: Barrier price level
            barrier_type: Type of barrier ('up-and-out', 'up-and-in',
                          'down-and-out', 'down-and-in')
            option_type: 'call' or 'put'
            n_simulations: Number of simulations
            n_timesteps: Number of timesteps for monitoring
            rebate: Rebate paid if option is knocked out

        Returns:
            MonteCarloResult with price and statistics
        """
        paths, _ = self._generate_paths(n_simulations, n_timesteps)
        terminal_prices = paths[:, -1]

        # Determine barrier crossing
        max_prices = np.max(paths, axis=1)
        min_prices = np.min(paths, axis=1)

        if barrier_type == 'up-and-out':
            crossed = max_prices >= barrier
            active = ~crossed
        elif barrier_type == 'up-and-in':
            crossed = max_prices >= barrier
            active = crossed
        elif barrier_type == 'down-and-out':
            crossed = min_prices <= barrier
            active = ~crossed
        elif barrier_type == 'down-and-in':
            crossed = min_prices <= barrier
            active = crossed
        else:
            raise ValueError(f"Unknown barrier_type: {barrier_type}")

        # Calculate vanilla payoff
        if option_type == 'call':
            vanilla_payoff = np.maximum(terminal_prices - self.K, 0)
        else:
            vanilla_payoff = np.maximum(self.K - terminal_prices, 0)

        # Apply barrier condition
        payoffs = np.where(active, vanilla_payoff, rebate)

        discount = np.exp(-self.r * self.T)
        discounted_payoffs = discount * payoffs

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)

        return MonteCarloResult(
            price=price,
            standard_error=std_error,
            confidence_interval_95=ci_95,
            n_simulations=n_simulations,
            n_timesteps=n_timesteps,
            convergence_achieved=std_error < 0.01 * price if price > 0 else True,
            exercise_probability=float(np.mean(active))
        )

    def price_lookback_option(
        self,
        option_type: str = 'call',  # 'call' or 'put'
        strike_type: str = 'floating',  # 'floating' or 'fixed'
        n_simulations: int = 100000,
        n_timesteps: int = 252
    ) -> MonteCarloResult:
        """
        Price lookback option using Monte Carlo.

        Lookback options have payoff depending on the maximum or minimum
        price achieved during the option's life, eliminating timing risk.

        Args:
            option_type: 'call' or 'put'
            strike_type: 'floating' (strike = min/max) or 'fixed'
            n_simulations: Number of simulations
            n_timesteps: Number of monitoring points

        Returns:
            MonteCarloResult with price and statistics
        """
        paths, _ = self._generate_paths(n_simulations, n_timesteps)
        terminal_prices = paths[:, -1]
        max_prices = np.max(paths, axis=1)
        min_prices = np.min(paths, axis=1)

        if strike_type == 'floating':
            # Floating strike lookback
            if option_type == 'call':
                # Right to buy at the lowest price observed
                payoffs = terminal_prices - min_prices
            else:
                # Right to sell at the highest price observed
                payoffs = max_prices - terminal_prices
        else:
            # Fixed strike lookback
            if option_type == 'call':
                # Payoff based on maximum price
                payoffs = np.maximum(max_prices - self.K, 0)
            else:
                # Payoff based on minimum price
                payoffs = np.maximum(self.K - min_prices, 0)

        discount = np.exp(-self.r * self.T)
        discounted_payoffs = discount * payoffs

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)

        return MonteCarloResult(
            price=price,
            standard_error=std_error,
            confidence_interval_95=ci_95,
            n_simulations=n_simulations,
            n_timesteps=n_timesteps,
            convergence_achieved=std_error < 0.01 * price if price > 0 else True
        )

    def price_american_option_lsm(
        self,
        option_type: str = 'put',  # American puts are more common
        n_simulations: int = 50000,
        n_timesteps: int = 52,  # Weekly monitoring
        polynomial_degree: int = 3
    ) -> MonteCarloResult:
        """
        Price American option using Longstaff-Schwartz Least Squares Monte Carlo.

        The LSM algorithm estimates the continuation value at each exercise
        date by regressing discounted future payoffs on polynomial basis
        functions of the current price.

        ★ Insight ─────────────────────────────────────
        American options can be exercised early, so we need to estimate
        the expected continuation value at each point. LSM uses regression
        to approximate this value, then compares with immediate exercise.
        ─────────────────────────────────────────────────

        Args:
            option_type: 'call' or 'put'
            n_simulations: Number of simulation paths
            n_timesteps: Number of exercise opportunities
            polynomial_degree: Degree of polynomial for regression

        Returns:
            MonteCarloResult with American option price
        """
        dt = self.T / n_timesteps
        discount_factor = np.exp(-self.r * dt)

        # Generate paths
        paths, _ = self._generate_paths(n_simulations, n_timesteps, antithetic=False)

        # Calculate intrinsic value at each time step
        if option_type == 'call':
            intrinsic = np.maximum(paths - self.K, 0)
        else:
            intrinsic = np.maximum(self.K - paths, 0)

        # Initialize value array with terminal payoff
        cash_flows = intrinsic[:, -1].copy()
        exercise_times = np.full(n_simulations, n_timesteps)

        # Backward induction
        for t in range(n_timesteps - 1, 0, -1):
            # Paths that are in-the-money at time t
            itm = intrinsic[:, t] > 0
            n_itm = np.sum(itm)

            if n_itm == 0:
                continue

            # Prices and continuation values for ITM paths
            S_itm = paths[itm, t]
            cash_flows_itm = cash_flows[itm] * discount_factor

            # Polynomial regression for continuation value
            # Basis: 1, S, S^2, S^3, ...
            basis = np.column_stack([
                S_itm ** p for p in range(polynomial_degree + 1)
            ])

            # Least squares regression
            try:
                coeffs = np.linalg.lstsq(basis, cash_flows_itm, rcond=None)[0]
                continuation_value = basis @ coeffs
            except np.linalg.LinAlgError:
                continuation_value = cash_flows_itm

            # Exercise decision: exercise if intrinsic > continuation
            exercise = intrinsic[itm, t] > continuation_value

            # Update cash flows for paths that exercise
            itm_indices = np.where(itm)[0]
            exercise_indices = itm_indices[exercise]

            cash_flows[exercise_indices] = intrinsic[exercise_indices, t]
            exercise_times[exercise_indices] = t

            # Discount remaining cash flows
            cash_flows = cash_flows * discount_factor

        # Final discount from t=1 to t=0
        cash_flows = cash_flows * discount_factor

        price = np.mean(cash_flows)
        std_error = np.std(cash_flows) / np.sqrt(n_simulations)
        ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)

        # Calculate early exercise probability
        early_exercise_prob = np.mean(exercise_times < n_timesteps)

        # Compare with European price for early exercise premium
        if option_type == 'call':
            euro_result = self.price_european_call(n_simulations, n_timesteps)
        else:
            euro_result = self.price_european_put(n_simulations, n_timesteps)

        early_exercise_premium = price - euro_result.price

        return MonteCarloResult(
            price=price,
            standard_error=std_error,
            confidence_interval_95=ci_95,
            n_simulations=n_simulations,
            n_timesteps=n_timesteps,
            convergence_achieved=std_error < 0.01 * price if price > 0 else True,
            exercise_probability=early_exercise_prob,
            early_exercise_premium=early_exercise_premium
        )

    def validate_against_black_scholes(
        self,
        n_simulations: int = 100000,
        n_timesteps: int = 252
    ) -> Dict[str, float]:
        """
        Validate Monte Carlo pricing against Black-Scholes closed-form.

        Useful for verifying the Monte Carlo implementation is correct.

        Returns:
            Dictionary with BS price, MC price, and error
        """
        # Black-Scholes prices
        bs_call = OptionsGreeks.black_scholes_call(
            self.S, self.K, self.T, self.r, self.sigma, self.q
        )
        bs_put = OptionsGreeks.black_scholes_put(
            self.S, self.K, self.T, self.r, self.sigma, self.q
        )

        # Monte Carlo prices
        mc_call = self.price_european_call(n_simulations, n_timesteps)
        mc_put = self.price_european_put(n_simulations, n_timesteps)

        return {
            'bs_call': bs_call,
            'mc_call': mc_call.price,
            'mc_call_se': mc_call.standard_error,
            'call_error_pct': abs(mc_call.price - bs_call) / bs_call * 100 if bs_call > 0 else 0,
            'bs_put': bs_put,
            'mc_put': mc_put.price,
            'mc_put_se': mc_put.standard_error,
            'put_error_pct': abs(mc_put.price - bs_put) / bs_put * 100 if bs_put > 0 else 0
        }


# =============================================================================
# DIAGONAL AND RATIO SPREADS
# =============================================================================

@dataclass
class SpreadLeg:
    """
    Single leg of an options spread.

    Attributes:
        option_type: 'call' or 'put'
        strike: Strike price
        expiration: Days to expiration
        quantity: Number of contracts (positive = long, negative = short)
        premium: Premium paid/received per contract
    """
    option_type: str  # 'call' or 'put'
    strike: float
    expiration: float  # Time to expiration in years
    quantity: int  # Positive for long, negative for short
    premium: Optional[float] = None


@dataclass
class SpreadAnalysis:
    """
    Analysis results for an options spread.

    Attributes:
        net_premium: Net premium paid (positive) or received (negative)
        max_profit: Maximum possible profit
        max_loss: Maximum possible loss
        breakeven_points: List of breakeven prices
        profit_probability: Estimated probability of profit
        greeks: Net Greeks for the spread
        payoff_at_expiry: Dictionary mapping price to payoff
    """
    net_premium: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    profit_probability: Optional[float]
    greeks: Dict[str, float]
    payoff_at_expiry: Dict[float, float]


class OptionsSpreadAnalyzer:
    """
    Analyzer for complex options spreads including diagonal and ratio spreads.

    Supports:
    - Diagonal spreads (different strikes AND different expirations)
    - Ratio spreads (unequal quantities of long/short options)
    - Calendar spreads (same strike, different expirations)
    - Vertical spreads (same expiration, different strikes)
    - Back spreads and front spreads
    - Custom multi-leg strategies

    ★ Insight ─────────────────────────────────────
    Diagonal spreads combine features of calendar and vertical spreads:
    - Different strikes: creates a directional bias
    - Different expirations: exploits time decay differential
    Ratio spreads create asymmetric payoffs with unlimited risk on one side
    ─────────────────────────────────────────────────

    Example:
        >>> analyzer = OptionsSpreadAnalyzer(S=100, r=0.05, sigma=0.20)
        >>> # Diagonal call spread
        >>> result = analyzer.diagonal_spread(
        ...     near_strike=100, far_strike=105,
        ...     near_expiry=30, far_expiry=60
        ... )
        >>> print(f"Max profit: ${result.max_profit:.2f}")

    References:
        - Natenberg, S. "Option Volatility and Pricing" (2nd Ed)
        - McMillan, L. "Options as a Strategic Investment" (5th Ed)
    """

    def __init__(
        self,
        S: float,
        r: float = 0.05,
        sigma: float = 0.20,
        q: float = 0.0
    ):
        """
        Initialize the spread analyzer.

        Args:
            S: Current underlying price
            r: Risk-free interest rate
            sigma: Implied volatility (can be overridden per leg)
            q: Dividend yield
        """
        self.S = S
        self.r = r
        self.sigma = sigma
        self.q = q

    def _calculate_premium(
        self,
        strike: float,
        expiration: float,
        option_type: str,
        sigma: Optional[float] = None
    ) -> float:
        """Calculate option premium using Black-Scholes."""
        vol = sigma if sigma is not None else self.sigma

        if option_type == 'call':
            return OptionsGreeks.black_scholes_call(
                self.S, strike, expiration, self.r, vol, self.q
            )
        else:
            return OptionsGreeks.black_scholes_put(
                self.S, strike, expiration, self.r, vol, self.q
            )

    def _calculate_greeks(
        self,
        strike: float,
        expiration: float,
        option_type: str,
        sigma: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate Greeks for a single option."""
        vol = sigma if sigma is not None else self.sigma

        og = OptionsGreeks(
            spot=self.S,
            strike=strike,
            rate=self.r,
            time_to_expiry=expiration,
            volatility=vol,
            dividend_yield=self.q
        )

        return og.calculate_all_greeks(option_type=option_type).to_dict()

    def _payoff_at_expiry(
        self,
        legs: List[SpreadLeg],
        price_range: Optional[Tuple[float, float]] = None,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate payoff diagram for spread at expiration of near-term leg.

        Args:
            legs: List of SpreadLeg objects
            price_range: Optional (min, max) price range
            n_points: Number of price points

        Returns:
            Tuple of (prices, payoffs)
        """
        if price_range is None:
            all_strikes = [leg.strike for leg in legs]
            min_strike = min(all_strikes)
            max_strike = max(all_strikes)
            margin = 0.3 * (max_strike - min_strike + self.S * 0.1)
            price_range = (min_strike - margin, max_strike + margin)

        prices = np.linspace(price_range[0], price_range[1], n_points)
        payoffs = np.zeros(n_points)

        for leg in legs:
            premium = leg.premium if leg.premium is not None else \
                self._calculate_premium(leg.strike, leg.expiration, leg.option_type)

            for i, price in enumerate(prices):
                if leg.option_type == 'call':
                    intrinsic = max(0, price - leg.strike)
                else:
                    intrinsic = max(0, leg.strike - price)

                # For long positions: payoff = intrinsic - premium
                # For short positions: payoff = premium - intrinsic
                leg_payoff = leg.quantity * (intrinsic - premium)
                payoffs[i] += leg_payoff

        return prices, payoffs

    def diagonal_spread(
        self,
        near_strike: float,
        far_strike: float,
        near_expiry: float,
        far_expiry: float,
        option_type: str = 'call',
        near_vol: Optional[float] = None,
        far_vol: Optional[float] = None,
        is_debit: bool = True
    ) -> SpreadAnalysis:
        """
        Analyze a diagonal spread.

        A diagonal spread involves buying and selling options with both
        different strikes AND different expirations. It's a hybrid of
        calendar and vertical spreads.

        Diagonal Call Spread (Bullish):
            - Buy longer-dated call at lower strike
            - Sell shorter-dated call at higher strike

        Diagonal Put Spread (Bearish):
            - Buy longer-dated put at higher strike
            - Sell shorter-dated put at lower strike

        Args:
            near_strike: Strike of short (near-term) option
            far_strike: Strike of long (far-term) option
            near_expiry: Days to expiry for short option
            far_expiry: Days to expiry for long option
            option_type: 'call' or 'put'
            near_vol: Optional IV for near-term option
            far_vol: Optional IV for far-term option
            is_debit: If True, constructs as debit spread (buy far, sell near)

        Returns:
            SpreadAnalysis with complete spread metrics

        Example:
            >>> # Bullish diagonal call spread
            >>> result = analyzer.diagonal_spread(
            ...     near_strike=105, far_strike=100,
            ...     near_expiry=0.0833, far_expiry=0.25,  # 1 mo vs 3 mo
            ...     option_type='call'
            ... )
        """
        # Convert days to years if needed (assume input > 10 means days)
        if near_expiry > 10:
            near_expiry = near_expiry / 365
        if far_expiry > 10:
            far_expiry = far_expiry / 365

        # Calculate premiums
        far_premium = self._calculate_premium(far_strike, far_expiry, option_type, far_vol)
        near_premium = self._calculate_premium(near_strike, near_expiry, option_type, near_vol)

        # Construct legs based on direction
        if is_debit:
            # Debit diagonal: buy far, sell near
            legs = [
                SpreadLeg(option_type, far_strike, far_expiry, 1, far_premium),
                SpreadLeg(option_type, near_strike, near_expiry, -1, near_premium)
            ]
            net_premium = far_premium - near_premium
        else:
            # Credit diagonal: sell far, buy near
            legs = [
                SpreadLeg(option_type, far_strike, far_expiry, -1, far_premium),
                SpreadLeg(option_type, near_strike, near_expiry, 1, near_premium)
            ]
            net_premium = near_premium - far_premium

        # Calculate Greeks (at near-term expiry)
        far_greeks = self._calculate_greeks(far_strike, far_expiry - near_expiry, option_type, far_vol)
        near_greeks = self._calculate_greeks(near_strike, near_expiry, option_type, near_vol)

        net_greeks = {
            'delta': legs[0].quantity * far_greeks.get('delta', 0) + legs[1].quantity * near_greeks.get('delta', 0),
            'gamma': legs[0].quantity * far_greeks.get('gamma', 0) + legs[1].quantity * near_greeks.get('gamma', 0),
            'theta': legs[0].quantity * far_greeks.get('theta', 0) + legs[1].quantity * near_greeks.get('theta', 0),
            'vega': legs[0].quantity * far_greeks.get('vega', 0) + legs[1].quantity * near_greeks.get('vega', 0),
        }

        # Payoff at near-term expiry
        prices, payoffs = self._payoff_at_expiry(legs)

        # Calculate max profit/loss
        max_profit = float(np.max(payoffs))
        max_loss = float(np.min(payoffs))

        # Find breakeven points
        breakeven_points = []
        for i in range(1, len(payoffs)):
            if (payoffs[i-1] < 0 and payoffs[i] >= 0) or (payoffs[i-1] >= 0 and payoffs[i] < 0):
                # Linear interpolation
                ratio = -payoffs[i-1] / (payoffs[i] - payoffs[i-1])
                be = prices[i-1] + ratio * (prices[i] - prices[i-1])
                breakeven_points.append(float(be))

        # Estimate profit probability (assuming lognormal)
        profit_prob = None
        if len(breakeven_points) > 0:
            from scipy.stats import norm
            # Use lognormal approximation
            vol_annual = self.sigma
            drift = (self.r - self.q - 0.5 * vol_annual ** 2) * near_expiry
            vol_t = vol_annual * np.sqrt(near_expiry)

            if len(breakeven_points) == 1:
                be = breakeven_points[0]
                d = (np.log(be / self.S) - drift) / vol_t
                # For debit spread, profit if price above breakeven
                if option_type == 'call' and is_debit:
                    profit_prob = 1 - norm.cdf(d)
                else:
                    profit_prob = norm.cdf(d)

        payoff_dict = dict(zip(prices.tolist(), payoffs.tolist()))

        return SpreadAnalysis(
            net_premium=net_premium,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=breakeven_points,
            profit_probability=profit_prob,
            greeks=net_greeks,
            payoff_at_expiry=payoff_dict
        )

    def ratio_spread(
        self,
        long_strike: float,
        short_strike: float,
        expiration: float,
        long_qty: int = 1,
        short_qty: int = 2,
        option_type: str = 'call',
        long_vol: Optional[float] = None,
        short_vol: Optional[float] = None
    ) -> SpreadAnalysis:
        """
        Analyze a ratio spread.

        A ratio spread involves buying and selling different quantities
        of options. Common ratios are 1:2 or 1:3. These create asymmetric
        payoff profiles with potential unlimited risk on one side.

        Call Ratio Spread (Neutral to Bullish):
            - Buy 1 ITM/ATM call
            - Sell 2+ OTM calls
            Maximum profit at short strike, unlimited risk above

        Put Ratio Spread (Neutral to Bearish):
            - Buy 1 ITM/ATM put
            - Sell 2+ OTM puts
            Maximum profit at short strike, unlimited risk below

        ★ Insight ─────────────────────────────────────
        Ratio spreads are typically done for credits or small debits.
        The risk is unlimited beyond the short strikes, but the position
        profits from modest directional moves and time decay.
        ─────────────────────────────────────────────────

        Args:
            long_strike: Strike of long options
            short_strike: Strike of short options
            expiration: Days to expiration (or years if < 10)
            long_qty: Number of long options (positive)
            short_qty: Number of short options (positive, will be negated)
            option_type: 'call' or 'put'
            long_vol: Optional IV for long strike
            short_vol: Optional IV for short strike

        Returns:
            SpreadAnalysis with spread metrics

        Example:
            >>> # 1x2 Call Ratio Spread
            >>> result = analyzer.ratio_spread(
            ...     long_strike=100, short_strike=110,
            ...     expiration=30,
            ...     long_qty=1, short_qty=2,
            ...     option_type='call'
            ... )
        """
        # Convert days to years if needed
        if expiration > 10:
            expiration = expiration / 365

        # Calculate premiums
        long_premium = self._calculate_premium(long_strike, expiration, option_type, long_vol)
        short_premium = self._calculate_premium(short_strike, expiration, option_type, short_vol)

        # Construct legs
        legs = [
            SpreadLeg(option_type, long_strike, expiration, long_qty, long_premium),
            SpreadLeg(option_type, short_strike, expiration, -short_qty, short_premium)
        ]

        net_premium = long_qty * long_premium - short_qty * short_premium

        # Calculate Greeks
        long_greeks = self._calculate_greeks(long_strike, expiration, option_type, long_vol)
        short_greeks = self._calculate_greeks(short_strike, expiration, option_type, short_vol)

        net_greeks = {
            'delta': long_qty * long_greeks.get('delta', 0) - short_qty * short_greeks.get('delta', 0),
            'gamma': long_qty * long_greeks.get('gamma', 0) - short_qty * short_greeks.get('gamma', 0),
            'theta': long_qty * long_greeks.get('theta', 0) - short_qty * short_greeks.get('theta', 0),
            'vega': long_qty * long_greeks.get('vega', 0) - short_qty * short_greeks.get('vega', 0),
        }

        # Extended price range for unlimited risk visualization
        strike_diff = abs(short_strike - long_strike)
        if option_type == 'call':
            price_range = (long_strike - strike_diff * 2, short_strike + strike_diff * 4)
        else:
            price_range = (short_strike - strike_diff * 4, long_strike + strike_diff * 2)

        prices, payoffs = self._payoff_at_expiry(legs, price_range)

        max_profit = float(np.max(payoffs))
        max_loss = float(np.min(payoffs))  # Can be very negative (unlimited risk)

        # Find breakeven points
        breakeven_points = []
        for i in range(1, len(payoffs)):
            if (payoffs[i-1] < 0 and payoffs[i] >= 0) or (payoffs[i-1] >= 0 and payoffs[i] < 0):
                ratio = -payoffs[i-1] / (payoffs[i] - payoffs[i-1])
                be = prices[i-1] + ratio * (prices[i] - prices[i-1])
                breakeven_points.append(float(be))

        payoff_dict = dict(zip(prices.tolist(), payoffs.tolist()))

        return SpreadAnalysis(
            net_premium=net_premium,
            max_profit=max_profit,
            max_loss=max_loss,  # Often shows as very negative (unlimited)
            breakeven_points=breakeven_points,
            profit_probability=None,  # Complex to calculate for ratio spreads
            greeks=net_greeks,
            payoff_at_expiry=payoff_dict
        )

    def back_spread(
        self,
        atm_strike: float,
        otm_strike: float,
        expiration: float,
        ratio: int = 2,
        option_type: str = 'call'
    ) -> SpreadAnalysis:
        """
        Analyze a back spread (also called ratio back spread).

        Back spreads have unlimited profit potential in one direction.
        They're typically done for credits and profit from large moves.

        Call Back Spread (Long Volatility, Bullish):
            - Sell 1 ATM call
            - Buy 2 OTM calls
            Profits from large upside moves

        Put Back Spread (Long Volatility, Bearish):
            - Sell 1 ATM put
            - Buy 2 OTM puts
            Profits from large downside moves

        Args:
            atm_strike: Strike of short (ATM) option
            otm_strike: Strike of long (OTM) options
            expiration: Time to expiration
            ratio: Number of long options per short option
            option_type: 'call' or 'put'

        Returns:
            SpreadAnalysis with spread metrics
        """
        if expiration > 10:
            expiration = expiration / 365

        # Calculate premiums
        atm_premium = self._calculate_premium(atm_strike, expiration, option_type)
        otm_premium = self._calculate_premium(otm_strike, expiration, option_type)

        # Back spread: sell 1 ATM, buy multiple OTM
        legs = [
            SpreadLeg(option_type, atm_strike, expiration, -1, atm_premium),
            SpreadLeg(option_type, otm_strike, expiration, ratio, otm_premium)
        ]

        net_premium = atm_premium - ratio * otm_premium  # Usually positive (credit)

        # Calculate Greeks
        atm_greeks = self._calculate_greeks(atm_strike, expiration, option_type)
        otm_greeks = self._calculate_greeks(otm_strike, expiration, option_type)

        net_greeks = {
            'delta': -atm_greeks.get('delta', 0) + ratio * otm_greeks.get('delta', 0),
            'gamma': -atm_greeks.get('gamma', 0) + ratio * otm_greeks.get('gamma', 0),
            'theta': -atm_greeks.get('theta', 0) + ratio * otm_greeks.get('theta', 0),
            'vega': -atm_greeks.get('vega', 0) + ratio * otm_greeks.get('vega', 0),
        }

        # Extended price range
        strike_diff = abs(otm_strike - atm_strike)
        if option_type == 'call':
            price_range = (atm_strike - strike_diff * 2, otm_strike + strike_diff * 4)
        else:
            price_range = (otm_strike - strike_diff * 4, atm_strike + strike_diff * 2)

        prices, payoffs = self._payoff_at_expiry(legs, price_range)

        max_profit = float(np.max(payoffs))  # Can be very large (unlimited)
        max_loss = float(np.min(payoffs))

        # Find breakeven points
        breakeven_points = []
        for i in range(1, len(payoffs)):
            if (payoffs[i-1] < 0 and payoffs[i] >= 0) or (payoffs[i-1] >= 0 and payoffs[i] < 0):
                ratio_be = -payoffs[i-1] / (payoffs[i] - payoffs[i-1])
                be = prices[i-1] + ratio_be * (prices[i] - prices[i-1])
                breakeven_points.append(float(be))

        payoff_dict = dict(zip(prices.tolist(), payoffs.tolist()))

        return SpreadAnalysis(
            net_premium=net_premium,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=breakeven_points,
            profit_probability=None,
            greeks=net_greeks,
            payoff_at_expiry=payoff_dict
        )

    def calendar_spread(
        self,
        strike: float,
        near_expiry: float,
        far_expiry: float,
        option_type: str = 'call',
        near_vol: Optional[float] = None,
        far_vol: Optional[float] = None
    ) -> SpreadAnalysis:
        """
        Analyze a calendar spread (time spread).

        Calendar spreads profit from time decay differential between
        near-term and far-term options at the same strike.

        Long Calendar Spread:
            - Sell near-term option
            - Buy far-term option at same strike
            Profits from time decay and volatility increases

        Args:
            strike: Strike price (same for both legs)
            near_expiry: Near-term expiration
            far_expiry: Far-term expiration
            option_type: 'call' or 'put'
            near_vol: Optional IV for near-term option
            far_vol: Optional IV for far-term option

        Returns:
            SpreadAnalysis with spread metrics
        """
        # Use diagonal spread with same strike
        return self.diagonal_spread(
            near_strike=strike,
            far_strike=strike,
            near_expiry=near_expiry,
            far_expiry=far_expiry,
            option_type=option_type,
            near_vol=near_vol,
            far_vol=far_vol,
            is_debit=True
        )

    def analyze_custom_spread(
        self,
        legs: List[SpreadLeg]
    ) -> SpreadAnalysis:
        """
        Analyze any custom multi-leg options spread.

        Args:
            legs: List of SpreadLeg objects defining the strategy

        Returns:
            SpreadAnalysis with complete metrics

        Example:
            >>> legs = [
            ...     SpreadLeg('call', 100, 0.0833, 1, 5.0),   # Long 1 call
            ...     SpreadLeg('call', 105, 0.0833, -2, 2.5),  # Short 2 calls
            ...     SpreadLeg('call', 110, 0.0833, 1, 1.0)    # Long 1 call (butterfly)
            ... ]
            >>> result = analyzer.analyze_custom_spread(legs)
        """
        # Calculate premiums if not provided
        for leg in legs:
            if leg.premium is None:
                leg.premium = self._calculate_premium(
                    leg.strike, leg.expiration, leg.option_type
                )

        # Net premium
        net_premium = sum(leg.quantity * leg.premium for leg in legs)

        # Calculate net Greeks
        net_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

        for leg in legs:
            leg_greeks = self._calculate_greeks(leg.strike, leg.expiration, leg.option_type)
            for greek in net_greeks:
                net_greeks[greek] += leg.quantity * leg_greeks.get(greek, 0)

        # Payoff calculation
        prices, payoffs = self._payoff_at_expiry(legs)

        max_profit = float(np.max(payoffs))
        max_loss = float(np.min(payoffs))

        # Find breakeven points
        breakeven_points = []
        for i in range(1, len(payoffs)):
            if (payoffs[i-1] < 0 and payoffs[i] >= 0) or (payoffs[i-1] >= 0 and payoffs[i] < 0):
                ratio = -payoffs[i-1] / (payoffs[i] - payoffs[i-1])
                be = prices[i-1] + ratio * (prices[i] - prices[i-1])
                breakeven_points.append(float(be))

        payoff_dict = dict(zip(prices.tolist(), payoffs.tolist()))

        return SpreadAnalysis(
            net_premium=net_premium,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=breakeven_points,
            profit_probability=None,
            greeks=net_greeks,
            payoff_at_expiry=payoff_dict
        )

    def compare_spreads(
        self,
        spreads: Dict[str, SpreadAnalysis]
    ) -> pd.DataFrame:
        """
        Compare multiple spread strategies side by side.

        Args:
            spreads: Dictionary mapping strategy name to SpreadAnalysis

        Returns:
            DataFrame comparing key metrics
        """
        comparison = []

        for name, analysis in spreads.items():
            comparison.append({
                'Strategy': name,
                'Net Premium': f"${analysis.net_premium:.2f}",
                'Max Profit': f"${analysis.max_profit:.2f}" if analysis.max_profit < 1e6 else 'Unlimited',
                'Max Loss': f"${analysis.max_loss:.2f}" if analysis.max_loss > -1e6 else 'Unlimited',
                'Risk/Reward': f"{abs(analysis.max_loss / analysis.max_profit):.2f}" if analysis.max_profit != 0 and abs(analysis.max_loss) < 1e6 else 'N/A',
                'Breakevens': ', '.join([f"${be:.2f}" for be in analysis.breakeven_points]),
                'Delta': f"{analysis.greeks.get('delta', 0):.3f}",
                'Theta': f"{analysis.greeks.get('theta', 0):.3f}",
                'Vega': f"{analysis.greeks.get('vega', 0):.3f}"
            })

        return pd.DataFrame(comparison)
