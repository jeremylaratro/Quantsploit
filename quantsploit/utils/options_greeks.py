"""
Options Greeks Calculator

This module provides functions to calculate options Greeks using
the Black-Scholes model and other pricing models.

Greeks calculated:
- Delta: Rate of change of option price with respect to underlying price
- Gamma: Rate of change of delta with respect to underlying price
- Theta: Rate of change of option price with respect to time
- Vega: Rate of change of option price with respect to volatility
- Rho: Rate of change of option price with respect to interest rate

Additional calculations:
- Implied Volatility using Newton-Raphson method
- Black-Scholes option pricing
- Probability of profit
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


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

        # Probability of profit (simplified)
        if option_type.lower() == "call":
            breakeven = K + price
            prob_profit = 1 - norm.cdf((np.log(breakeven / S)) / (sigma * np.sqrt(T)))
        else:
            breakeven = K - price
            prob_profit = norm.cdf((np.log(breakeven / S)) / (sigma * np.sqrt(T)))

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
