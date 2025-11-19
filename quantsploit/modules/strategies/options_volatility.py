"""
Options Volatility Trading Strategies

This module implements advanced options strategies that profit from volatility:

1. Long Straddle: Profit from large price moves in either direction
2. Short Straddle: Profit from low volatility (range-bound)
3. Long Strangle: Cheaper alternative to straddle
4. Short Strangle: Selling OTM options for premium
5. IV Rank Analysis: Compare current IV to historical range
6. Volatility Arbitrage: Trade IV vs HV discrepancies

State-of-the-art techniques:
- Implied Volatility (IV) analysis and ranking
- Historical Volatility (HV) comparison
- Volatility smile/skew analysis
- Greeks-based position management
- Expected move calculations
"""

from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher
from quantsploit.utils.options_greeks import (
    OptionsGreeks, calculate_historical_volatility, calculate_option_profit_loss
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class OptionsVolatilityStrategy(BaseModule):
    """
    Options Volatility Trading Strategies

    Implements various volatility-based options strategies
    """

    @property
    def name(self) -> str:


        return "options_volatility"


    @property
    def description(self) -> str:


        return "Options strategies based on volatility analysis"


    @property
    def author(self) -> str:


        return "Quantsploit Team"


    @property
    def category(self) -> str:


        return "strategy"

    def _init_options(self):
        super()._init_options()
        self.options.update({
        "SYMBOL": {
            "description": "Stock symbol to analyze",
            "required": True,
            "value": "AAPL"
        },
        "STRATEGY": {
            "description": "Strategy: long_straddle, short_straddle, long_strangle, short_strangle, iv_rank",
            "required": False,
            "value": "long_straddle"
        },
        "DAYS_TO_EXPIRATION": {
            "description": "Target days to expiration",
            "required": False,
            "value": 30
        },
        "RISK_FREE_RATE": {
            "description": "Risk-free interest rate (annual)",
            "required": False,
            "value": 0.05
        },
        "DIVIDEND_YIELD": {
            "description": "Dividend yield (annual)",
            "required": False,
            "value": 0.0
        },
        "IV_PERCENTILE_THRESHOLD": {
            "description": "IV percentile threshold for entry (e.g., 50 = median)",
            "required": False,
            "value": 50
        },
        "HV_WINDOW": {
            "description": "Historical volatility lookback window (days)",
            "required": False,
            "value": 30
        },
        "STRANGLE_OTM_PCT": {
            "description": "OTM percentage for strangle strikes (e.g., 5 = 5%)",
            "required": False,
            "value": 5
        },
        "CONTRACTS": {
            "description": "Number of option contracts to trade",
            "required": False,
            "value": 1
        },
        })


    def calculate_iv_rank(
        self,
        current_iv: float,
        historical_ivs: List[float]
    ) -> float:
        """
        Calculate IV Rank (current IV's percentile in historical IV range)

        Args:
            current_iv: Current implied volatility
            historical_ivs: List of historical IVs

        Returns:
            IV Rank (0-100)
        """
        if not historical_ivs:
            return 50.0

        iv_min = min(historical_ivs)
        iv_max = max(historical_ivs)

        if iv_max == iv_min:
            return 50.0

        iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100

        return iv_rank

    def calculate_iv_percentile(
        self,
        current_iv: float,
        historical_ivs: List[float]
    ) -> float:
        """
        Calculate IV Percentile (percentage of days where IV was below current)

        Args:
            current_iv: Current implied volatility
            historical_ivs: List of historical IVs

        Returns:
            IV Percentile (0-100)
        """
        if not historical_ivs:
            return 50.0

        below_count = sum(1 for iv in historical_ivs if iv < current_iv)
        iv_percentile = (below_count / len(historical_ivs)) * 100

        return iv_percentile

    def long_straddle_analysis(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> Dict:
        """
        Analyze long straddle strategy (buy ATM call + put)

        Args:
            S: Current stock price
            K: Strike price (typically ATM)
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Implied volatility
            q: Dividend yield

        Returns:
            Dictionary with strategy analysis
        """
        # Calculate call and put prices and Greeks
        call_greeks = OptionsGreeks.calculate_all_greeks(S, K, T, r, sigma, "call", q)
        put_greeks = OptionsGreeks.calculate_all_greeks(S, K, T, r, sigma, "put", q)

        # Straddle metrics
        total_premium = call_greeks['price'] + put_greeks['price']
        total_delta = call_greeks['delta'] + put_greeks['delta']  # Should be ~0 for ATM
        total_gamma = call_greeks['gamma'] + put_greeks['gamma']
        total_theta = call_greeks['theta'] + put_greeks['theta']
        total_vega = call_greeks['vega'] + put_greeks['vega']

        # Breakeven points
        upper_breakeven = K + total_premium
        lower_breakeven = K - total_premium

        # Expected move (1 standard deviation)
        expected_move = S * sigma * np.sqrt(T)

        # Profit potential
        max_loss = total_premium  # If stock stays at strike
        max_profit = "Unlimited"  # Large moves in either direction

        # Probability of profit (simplified - need to move beyond breakeven)
        prob_above_upper = 1 - norm.cdf((np.log(upper_breakeven / S)) / (sigma * np.sqrt(T)))
        prob_below_lower = norm.cdf((np.log(lower_breakeven / S)) / (sigma * np.sqrt(T)))
        prob_profit = (prob_above_upper + prob_below_lower) * 100

        from scipy.stats import norm

        return {
            "strategy": "Long Straddle",
            "strike": K,
            "call_premium": call_greeks['price'],
            "put_premium": put_greeks['price'],
            "total_premium": total_premium,
            "upper_breakeven": upper_breakeven,
            "lower_breakeven": lower_breakeven,
            "expected_move": expected_move,
            "max_loss": max_loss,
            "max_profit": max_profit,
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_theta": total_theta,
            "total_vega": total_vega,
            "probability_of_profit": prob_profit,
            "days_to_expiration": T * 365,
            "implied_volatility": sigma,
            "recommendation": "Enter if expecting large move or IV is low"
        }

    def short_straddle_analysis(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> Dict:
        """
        Analyze short straddle strategy (sell ATM call + put)

        High risk, high reward - profit from low volatility
        """
        # Calculate long straddle first
        long_analysis = self.long_straddle_analysis(S, K, T, r, sigma, q)

        # Short straddle is opposite
        total_credit = long_analysis['total_premium']

        return {
            "strategy": "Short Straddle",
            "strike": K,
            "call_premium": long_analysis['call_premium'],
            "put_premium": long_analysis['put_premium'],
            "total_credit": total_credit,
            "upper_breakeven": long_analysis['upper_breakeven'],
            "lower_breakeven": long_analysis['lower_breakeven'],
            "expected_move": long_analysis['expected_move'],
            "max_profit": total_credit,
            "max_loss": "Unlimited",
            "total_delta": -long_analysis['total_delta'],
            "total_gamma": -long_analysis['total_gamma'],
            "total_theta": -long_analysis['total_theta'],
            "total_vega": -long_analysis['total_vega'],
            "probability_of_profit": 100 - long_analysis['probability_of_profit'],
            "days_to_expiration": T * 365,
            "implied_volatility": sigma,
            "recommendation": "Enter if expecting low volatility or IV is high"
        }

    def long_strangle_analysis(
        self,
        S: float,
        K_call: float,
        K_put: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> Dict:
        """
        Analyze long strangle strategy (buy OTM call + OTM put)

        Cheaper than straddle but requires larger move
        """
        # Calculate call and put prices and Greeks
        call_greeks = OptionsGreeks.calculate_all_greeks(S, K_call, T, r, sigma, "call", q)
        put_greeks = OptionsGreeks.calculate_all_greeks(S, K_put, T, r, sigma, "put", q)

        # Strangle metrics
        total_premium = call_greeks['price'] + put_greeks['price']
        total_delta = call_greeks['delta'] + put_greeks['delta']
        total_gamma = call_greeks['gamma'] + put_greeks['gamma']
        total_theta = call_greeks['theta'] + put_greeks['theta']
        total_vega = call_greeks['vega'] + put_greeks['vega']

        # Breakeven points
        upper_breakeven = K_call + total_premium
        lower_breakeven = K_put - total_premium

        # Expected move
        expected_move = S * sigma * np.sqrt(T)

        # Max loss and profit
        max_loss = total_premium
        max_profit = "Unlimited"

        from scipy.stats import norm

        # Probability of profit
        prob_above_upper = 1 - norm.cdf((np.log(upper_breakeven / S)) / (sigma * np.sqrt(T)))
        prob_below_lower = norm.cdf((np.log(lower_breakeven / S)) / (sigma * np.sqrt(T)))
        prob_profit = (prob_above_upper + prob_below_lower) * 100

        return {
            "strategy": "Long Strangle",
            "call_strike": K_call,
            "put_strike": K_put,
            "call_premium": call_greeks['price'],
            "put_premium": put_greeks['price'],
            "total_premium": total_premium,
            "upper_breakeven": upper_breakeven,
            "lower_breakeven": lower_breakeven,
            "expected_move": expected_move,
            "max_loss": max_loss,
            "max_profit": max_profit,
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_theta": total_theta,
            "total_vega": total_vega,
            "probability_of_profit": prob_profit,
            "days_to_expiration": T * 365,
            "implied_volatility": sigma,
            "recommendation": "Cheaper than straddle, enter if expecting large move"
        }

    def iv_rank_analysis(
        self,
        df: pd.DataFrame,
        current_iv: float,
        hv_window: int
    ) -> Dict:
        """
        Analyze IV rank and compare to historical volatility

        Args:
            df: Price data
            current_iv: Current implied volatility
            hv_window: Window for HV calculation

        Returns:
            Dictionary with IV analysis
        """
        # Calculate historical volatility
        prices = df['Close'].values
        hv = calculate_historical_volatility(prices, hv_window)

        # Calculate rolling IVs (approximation using HV)
        # In practice, you'd use actual historical IV data
        rolling_hvs = []
        for i in range(hv_window, len(prices)):
            window_hv = calculate_historical_volatility(prices[:i], hv_window)
            rolling_hvs.append(window_hv)

        # Calculate IV rank and percentile
        iv_rank = self.calculate_iv_rank(current_iv, rolling_hvs)
        iv_percentile = self.calculate_iv_percentile(current_iv, rolling_hvs)

        # IV vs HV comparison
        iv_hv_ratio = current_iv / hv if hv > 0 else 1.0

        # Determine recommendation
        if iv_rank > 75:
            recommendation = "HIGH IV - Consider selling options (short straddle/strangle)"
            regime = "HIGH"
        elif iv_rank < 25:
            recommendation = "LOW IV - Consider buying options (long straddle/strangle)"
            regime = "LOW"
        else:
            recommendation = "NEUTRAL IV - Wait for better setup"
            regime = "NEUTRAL"

        return {
            "current_iv": current_iv,
            "historical_volatility": hv,
            "iv_rank": iv_rank,
            "iv_percentile": iv_percentile,
            "iv_hv_ratio": iv_hv_ratio,
            "iv_regime": regime,
            "recommendation": recommendation,
            "hv_window": hv_window
        }

    def run(self):
        """Execute the options volatility strategy"""
        symbol = self.options["SYMBOL"]["value"]
        strategy = self.options["STRATEGY"]["value"]
        dte = int(self.options["DAYS_TO_EXPIRATION"]["value"])
        risk_free_rate = float(self.options["RISK_FREE_RATE"]["value"])
        dividend_yield = float(self.options["DIVIDEND_YIELD"]["value"])
        iv_threshold = float(self.options["IV_PERCENTILE_THRESHOLD"]["value"])
        hv_window = int(self.options["HV_WINDOW"]["value"])
        strangle_otm_pct = float(self.options["STRANGLE_OTM_PCT"]["value"]) / 100
        contracts = int(self.options["CONTRACTS"]["value"])

        self.print_status(f"Running Options Volatility Strategy: {strategy}")
        self.print_info(f"Symbol: {symbol}")

        # Fetch stock data
        data_fetcher = DataFetcher(self.database)
        df = data_fetcher.get_stock_data(symbol, period="1y", interval="1d")

        if df is None or len(df) < hv_window:
            self.print_error("Insufficient data for analysis")
            return {"error": "Insufficient data"}

        current_price = df['Close'].iloc[-1]
        self.print_info(f"Current Price: ${current_price:.2f}")

        # Calculate historical volatility
        hv = calculate_historical_volatility(df['Close'].values, hv_window)
        self.print_info(f"Historical Volatility ({hv_window}d): {hv:.2%}")

        # Estimate current IV (in practice, fetch from options chain)
        # For now, use HV as a proxy
        current_iv = hv * 1.2  # IV is typically higher than HV

        self.print_info(f"Estimated Implied Volatility: {current_iv:.2%}")

        # Convert DTE to years
        T = dte / 365.0

        # Execute strategy analysis
        if strategy == "long_straddle":
            # ATM strike
            K = round(current_price / 5) * 5  # Round to nearest $5

            self.print_status(f"\nAnalyzing Long Straddle (Strike: ${K})")

            analysis = self.long_straddle_analysis(
                current_price, K, T, risk_free_rate, current_iv, dividend_yield
            )

        elif strategy == "short_straddle":
            # ATM strike
            K = round(current_price / 5) * 5

            self.print_status(f"\nAnalyzing Short Straddle (Strike: ${K})")

            analysis = self.short_straddle_analysis(
                current_price, K, T, risk_free_rate, current_iv, dividend_yield
            )

        elif strategy == "long_strangle":
            # OTM strikes
            K_call = round(current_price * (1 + strangle_otm_pct) / 5) * 5
            K_put = round(current_price * (1 - strangle_otm_pct) / 5) * 5

            self.print_status(f"\nAnalyzing Long Strangle")
            self.print_info(f"Call Strike: ${K_call}")
            self.print_info(f"Put Strike: ${K_put}")

            analysis = self.long_strangle_analysis(
                current_price, K_call, K_put, T, risk_free_rate, current_iv, dividend_yield
            )

        elif strategy == "short_strangle":
            # OTM strikes
            K_call = round(current_price * (1 + strangle_otm_pct) / 5) * 5
            K_put = round(current_price * (1 - strangle_otm_pct) / 5) * 5

            self.print_status(f"\nAnalyzing Short Strangle")
            self.print_info(f"Call Strike: ${K_call}")
            self.print_info(f"Put Strike: ${K_put}")

            # Calculate long strangle first, then invert
            long_analysis = self.long_strangle_analysis(
                current_price, K_call, K_put, T, risk_free_rate, current_iv, dividend_yield
            )

            analysis = {
                "strategy": "Short Strangle",
                "call_strike": K_call,
                "put_strike": K_put,
                "call_premium": long_analysis['call_premium'],
                "put_premium": long_analysis['put_premium'],
                "total_credit": long_analysis['total_premium'],
                "upper_breakeven": long_analysis['upper_breakeven'],
                "lower_breakeven": long_analysis['lower_breakeven'],
                "expected_move": long_analysis['expected_move'],
                "max_profit": long_analysis['total_premium'],
                "max_loss": "Unlimited",
                "total_delta": -long_analysis['total_delta'],
                "total_gamma": -long_analysis['total_gamma'],
                "total_theta": -long_analysis['total_theta'],
                "total_vega": -long_analysis['total_vega'],
                "probability_of_profit": 100 - long_analysis['probability_of_profit'],
                "days_to_expiration": T * 365,
                "implied_volatility": current_iv,
                "recommendation": "Enter if expecting low volatility and IV is high"
            }

        elif strategy == "iv_rank":
            self.print_status("\nIV Rank Analysis")

            analysis = self.iv_rank_analysis(df, current_iv, hv_window)

        else:
            self.print_error(f"Unknown strategy: {strategy}")
            return {"error": f"Unknown strategy: {strategy}"}

        # Display results
        self.print_good("\n=== Strategy Analysis ===")

        for key, value in analysis.items():
            if isinstance(value, float):
                if key in ['implied_volatility', 'historical_volatility']:
                    self.print_info(f"{key}: {value:.2%}")
                elif 'probability' in key.lower() or 'rank' in key.lower() or 'percentile' in key.lower():
                    self.print_info(f"{key}: {value:.2f}%")
                else:
                    self.print_info(f"{key}: {value:.4f}")
            else:
                self.print_info(f"{key}: {value}")

        # Calculate position costs/credits
        if strategy != "iv_rank":
            multiplier = 100  # Options multiplier

            if "total_premium" in analysis:
                cost_per_contract = analysis['total_premium'] * multiplier
                total_cost = cost_per_contract * contracts

                self.print_info(f"\n=== Position Details ===")
                self.print_info(f"Contracts: {contracts}")
                self.print_info(f"Cost per contract: ${cost_per_contract:.2f}")
                self.print_info(f"Total cost: ${total_cost:.2f}")

            elif "total_credit" in analysis:
                credit_per_contract = analysis['total_credit'] * multiplier
                total_credit = credit_per_contract * contracts

                self.print_info(f"\n=== Position Details ===")
                self.print_info(f"Contracts: {contracts}")
                self.print_info(f"Credit per contract: ${credit_per_contract:.2f}")
                self.print_info(f"Total credit: ${total_credit:.2f}")

        self.print_good(f"\n=== Recommendation ===")
        self.print_info(analysis.get('recommendation', 'No recommendation available'))

        return {
            "symbol": symbol,
            "current_price": float(current_price),
            "strategy": strategy,
            "analysis": analysis,
            "contracts": contracts
        }
