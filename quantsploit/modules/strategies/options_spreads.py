"""
Advanced Options Spread Strategies

This module implements sophisticated options spread strategies:

1. Iron Condor: Neutral strategy for range-bound markets
2. Iron Butterfly: Similar to Iron Condor but tighter profit zone
3. Butterfly Spread: Limited risk, limited profit
4. Calendar Spread: Profit from time decay differences
5. Diagonal Spread: Combination of vertical and calendar
6. Credit Spreads: Bull put spread, bear call spread
7. Debit Spreads: Bull call spread, bear put spread

State-of-the-art techniques:
- Greeks-based position analysis
- Risk/reward optimization
- Probability of profit calculations
- Dynamic strike selection based on volatility
- Position adjustment recommendations
"""

from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher
from quantsploit.utils.options_greeks import (
    OptionsGreeks, calculate_historical_volatility
)
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class OptionsSpreadStrategy(BaseModule):
    """
    Advanced Options Spread Strategies

    Implements various options spread strategies with full analysis
    """

    @property


    def name(self) -> str:


        return "options_spreads"



    @property


    def description(self) -> str:


        return "Advanced options spreads: Iron Condor, Butterfly, Calendar, etc."



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
            "value": "SPY"
        },
        "STRATEGY": {
            "description": "Spread: iron_condor, iron_butterfly, butterfly, calendar, bull_call, bear_put",
            "required": False,
            "value": "iron_condor"
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
            "value": 0.02
        },
        "WING_WIDTH": {
            "description": "Width of wings for Iron Condor/Butterfly ($)",
            "required": False,
            "value": 5
        },
        "PROFIT_TARGET_PCT": {
            "description": "Profit target as % of max profit (50 = 50%)",
            "required": False,
            "value": 50
        },
        "CONTRACTS": {
            "description": "Number of contracts to trade",
            "required": False,
            "value": 1
        },
        })


    def iron_condor_analysis(
        self,
        S: float,
        wing_width: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> Dict:
        """
        Analyze Iron Condor spread

        Structure:
        - Sell OTM call
        - Buy further OTM call (protection)
        - Sell OTM put
        - Buy further OTM put (protection)

        Args:
            S: Current stock price
            wing_width: Width between short and long strikes
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Implied volatility
            q: Dividend yield

        Returns:
            Dictionary with strategy analysis
        """
        # Determine strikes based on expected move
        expected_move = S * sigma * np.sqrt(T)

        # Short strikes around +/- 1 std dev
        short_call_strike = round((S + expected_move) / 5) * 5
        short_put_strike = round((S - expected_move) / 5) * 5

        # Long strikes (wings)
        long_call_strike = short_call_strike + wing_width
        long_put_strike = short_put_strike - wing_width

        # Calculate premiums
        short_call = OptionsGreeks.calculate_all_greeks(S, short_call_strike, T, r, sigma, "call", q)
        long_call = OptionsGreeks.calculate_all_greeks(S, long_call_strike, T, r, sigma, "call", q)
        short_put = OptionsGreeks.calculate_all_greeks(S, short_put_strike, T, r, sigma, "put", q)
        long_put = OptionsGreeks.calculate_all_greeks(S, long_put_strike, T, r, sigma, "put", q)

        # Net credit received
        net_credit = (short_call['price'] - long_call['price'] +
                     short_put['price'] - long_put['price'])

        # Greeks
        total_delta = (short_call['delta'] * -1 + long_call['delta'] +
                      short_put['delta'] * -1 + long_put['delta'])
        total_gamma = (short_call['gamma'] * -1 + long_call['gamma'] +
                      short_put['gamma'] * -1 + long_put['gamma'])
        total_theta = (short_call['theta'] * -1 + long_call['theta'] +
                      short_put['theta'] * -1 + long_put['theta'])
        total_vega = (short_call['vega'] * -1 + long_call['vega'] +
                     short_put['vega'] * -1 + long_put['vega'])

        # Profit/Loss
        max_profit = net_credit
        max_loss = wing_width - net_credit

        # Breakeven points
        upper_breakeven = short_call_strike + net_credit
        lower_breakeven = short_put_strike - net_credit

        # Probability of profit (price stays between short strikes at expiration)
        prob_below_short_call = norm.cdf((np.log(short_call_strike / S)) / (sigma * np.sqrt(T)))
        prob_above_short_put = 1 - norm.cdf((np.log(short_put_strike / S)) / (sigma * np.sqrt(T)))
        prob_profit = (prob_below_short_call - (1 - prob_above_short_put)) * 100

        # Return on capital
        capital_required = max_loss  # Max loss is the capital at risk
        roc = (max_profit / capital_required) * 100 if capital_required > 0 else 0

        return {
            "strategy": "Iron Condor",
            "short_call_strike": short_call_strike,
            "long_call_strike": long_call_strike,
            "short_put_strike": short_put_strike,
            "long_put_strike": long_put_strike,
            "short_call_premium": short_call['price'],
            "long_call_premium": long_call['price'],
            "short_put_premium": short_put['price'],
            "long_put_premium": long_put['price'],
            "net_credit": net_credit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "upper_breakeven": upper_breakeven,
            "lower_breakeven": lower_breakeven,
            "profit_zone": f"${lower_breakeven:.2f} - ${upper_breakeven:.2f}",
            "total_delta": total_delta,
            "total_gamma": total_gamma,
            "total_theta": total_theta,
            "total_vega": total_vega,
            "probability_of_profit": prob_profit,
            "return_on_capital": roc,
            "days_to_expiration": T * 365,
            "expected_move": expected_move
        }

    def iron_butterfly_analysis(
        self,
        S: float,
        wing_width: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> Dict:
        """
        Analyze Iron Butterfly spread

        Structure:
        - Sell ATM call
        - Buy OTM call (protection)
        - Sell ATM put
        - Buy OTM put (protection)

        Higher credit but narrower profit zone than Iron Condor
        """
        # ATM strike
        atm_strike = round(S / 5) * 5

        # Wing strikes
        long_call_strike = atm_strike + wing_width
        long_put_strike = atm_strike - wing_width

        # Calculate premiums
        short_call = OptionsGreeks.calculate_all_greeks(S, atm_strike, T, r, sigma, "call", q)
        long_call = OptionsGreeks.calculate_all_greeks(S, long_call_strike, T, r, sigma, "call", q)
        short_put = OptionsGreeks.calculate_all_greeks(S, atm_strike, T, r, sigma, "put", q)
        long_put = OptionsGreeks.calculate_all_greeks(S, long_put_strike, T, r, sigma, "put", q)

        # Net credit
        net_credit = (short_call['price'] - long_call['price'] +
                     short_put['price'] - long_put['price'])

        # Greeks
        total_delta = (short_call['delta'] * -1 + long_call['delta'] +
                      short_put['delta'] * -1 + long_put['delta'])
        total_theta = (short_call['theta'] * -1 + long_call['theta'] +
                      short_put['theta'] * -1 + long_put['theta'])

        # Profit/Loss
        max_profit = net_credit
        max_loss = wing_width - net_credit

        # Breakeven
        upper_breakeven = atm_strike + net_credit
        lower_breakeven = atm_strike - net_credit

        # Probability
        prob_below_upper = norm.cdf((np.log(upper_breakeven / S)) / (sigma * np.sqrt(T)))
        prob_above_lower = 1 - norm.cdf((np.log(lower_breakeven / S)) / (sigma * np.sqrt(T)))
        prob_profit = (prob_below_upper - (1 - prob_above_lower)) * 100

        roc = (max_profit / max_loss) * 100 if max_loss > 0 else 0

        return {
            "strategy": "Iron Butterfly",
            "atm_strike": atm_strike,
            "long_call_strike": long_call_strike,
            "long_put_strike": long_put_strike,
            "net_credit": net_credit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "upper_breakeven": upper_breakeven,
            "lower_breakeven": lower_breakeven,
            "profit_zone": f"${lower_breakeven:.2f} - ${upper_breakeven:.2f}",
            "total_delta": total_delta,
            "total_theta": total_theta,
            "probability_of_profit": prob_profit,
            "return_on_capital": roc,
            "days_to_expiration": T * 365
        }

    def butterfly_spread_analysis(
        self,
        S: float,
        wing_width: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0
    ) -> Dict:
        """
        Analyze Butterfly spread (using calls or puts)

        Structure (Call Butterfly):
        - Buy 1 ITM call
        - Sell 2 ATM calls
        - Buy 1 OTM call

        Debit spread with limited risk and limited profit
        """
        # Strikes
        atm_strike = round(S / 5) * 5
        lower_strike = atm_strike - wing_width
        upper_strike = atm_strike + wing_width

        # Calculate premiums
        if option_type.lower() == "call":
            lower = OptionsGreeks.calculate_all_greeks(S, lower_strike, T, r, sigma, "call", q)
            middle = OptionsGreeks.calculate_all_greeks(S, atm_strike, T, r, sigma, "call", q)
            upper = OptionsGreeks.calculate_all_greeks(S, upper_strike, T, r, sigma, "call", q)
        else:
            lower = OptionsGreeks.calculate_all_greeks(S, lower_strike, T, r, sigma, "put", q)
            middle = OptionsGreeks.calculate_all_greeks(S, atm_strike, T, r, sigma, "put", q)
            upper = OptionsGreeks.calculate_all_greeks(S, upper_strike, T, r, sigma, "put", q)

        # Net debit
        net_debit = lower['price'] - 2 * middle['price'] + upper['price']

        # Profit/Loss
        max_profit = wing_width - net_debit
        max_loss = net_debit

        # Breakeven
        lower_breakeven = lower_strike + net_debit
        upper_breakeven = upper_strike - net_debit

        # Greeks
        total_delta = lower['delta'] - 2 * middle['delta'] + upper['delta']
        total_theta = lower['theta'] - 2 * middle['theta'] + upper['theta']

        roc = (max_profit / max_loss) * 100 if max_loss > 0 else 0

        return {
            "strategy": f"{option_type.title()} Butterfly Spread",
            "lower_strike": lower_strike,
            "middle_strike": atm_strike,
            "upper_strike": upper_strike,
            "net_debit": net_debit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "lower_breakeven": lower_breakeven,
            "upper_breakeven": upper_breakeven,
            "profit_zone": f"${lower_breakeven:.2f} - ${upper_breakeven:.2f}",
            "total_delta": total_delta,
            "total_theta": total_theta,
            "return_on_capital": roc,
            "days_to_expiration": T * 365
        }

    def calendar_spread_analysis(
        self,
        S: float,
        T_short: float,
        T_long: float,
        r: float,
        sigma: float,
        option_type: str = "call",
        q: float = 0.0
    ) -> Dict:
        """
        Analyze Calendar (Time) Spread

        Structure:
        - Sell near-term option
        - Buy longer-term option (same strike)

        Profits from time decay differential
        """
        # ATM strike
        strike = round(S / 5) * 5

        # Calculate premiums for different expirations
        short_option = OptionsGreeks.calculate_all_greeks(S, strike, T_short, r, sigma, option_type, q)
        long_option = OptionsGreeks.calculate_all_greeks(S, strike, T_long, r, sigma, option_type, q)

        # Net debit
        net_debit = long_option['price'] - short_option['price']

        # Greeks
        total_delta = long_option['delta'] - short_option['delta']
        total_theta = long_option['theta'] - short_option['theta']
        total_vega = long_option['vega'] - short_option['vega']

        return {
            "strategy": f"{option_type.title()} Calendar Spread",
            "strike": strike,
            "short_expiration_days": T_short * 365,
            "long_expiration_days": T_long * 365,
            "short_option_premium": short_option['price'],
            "long_option_premium": long_option['price'],
            "net_debit": net_debit,
            "max_loss": net_debit,
            "max_profit": "Varies based on volatility and price at near-term expiration",
            "total_delta": total_delta,
            "total_theta": total_theta,
            "total_vega": total_vega,
            "ideal_outcome": f"Stock price stays near ${strike:.2f} until short option expires"
        }

    def bull_call_spread_analysis(
        self,
        S: float,
        spread_width: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> Dict:
        """
        Analyze Bull Call Spread

        Structure:
        - Buy ATM or slightly OTM call
        - Sell higher strike call

        Bullish debit spread with limited risk and profit
        """
        # Strikes
        long_strike = round(S / 5) * 5
        short_strike = long_strike + spread_width

        # Calculate premiums
        long_call = OptionsGreeks.calculate_all_greeks(S, long_strike, T, r, sigma, "call", q)
        short_call = OptionsGreeks.calculate_all_greeks(S, short_strike, T, r, sigma, "call", q)

        # Net debit
        net_debit = long_call['price'] - short_call['price']

        # Profit/Loss
        max_profit = spread_width - net_debit
        max_loss = net_debit

        # Breakeven
        breakeven = long_strike + net_debit

        # Greeks
        total_delta = long_call['delta'] - short_call['delta']
        total_theta = long_call['theta'] - short_call['theta']

        # Probability
        prob_profit = (1 - norm.cdf((np.log(breakeven / S)) / (sigma * np.sqrt(T)))) * 100

        roc = (max_profit / max_loss) * 100 if max_loss > 0 else 0

        return {
            "strategy": "Bull Call Spread",
            "long_call_strike": long_strike,
            "short_call_strike": short_strike,
            "long_call_premium": long_call['price'],
            "short_call_premium": short_call['price'],
            "net_debit": net_debit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": breakeven,
            "total_delta": total_delta,
            "total_theta": total_theta,
            "probability_of_profit": prob_profit,
            "return_on_capital": roc,
            "days_to_expiration": T * 365
        }

    def bear_put_spread_analysis(
        self,
        S: float,
        spread_width: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> Dict:
        """
        Analyze Bear Put Spread

        Structure:
        - Buy ATM or slightly OTM put
        - Sell lower strike put

        Bearish debit spread with limited risk and profit
        """
        # Strikes
        long_strike = round(S / 5) * 5
        short_strike = long_strike - spread_width

        # Calculate premiums
        long_put = OptionsGreeks.calculate_all_greeks(S, long_strike, T, r, sigma, "put", q)
        short_put = OptionsGreeks.calculate_all_greeks(S, short_strike, T, r, sigma, "put", q)

        # Net debit
        net_debit = long_put['price'] - short_put['price']

        # Profit/Loss
        max_profit = spread_width - net_debit
        max_loss = net_debit

        # Breakeven
        breakeven = long_strike - net_debit

        # Greeks
        total_delta = long_put['delta'] - short_put['delta']
        total_theta = long_put['theta'] - short_put['theta']

        # Probability
        prob_profit = norm.cdf((np.log(breakeven / S)) / (sigma * np.sqrt(T))) * 100

        roc = (max_profit / max_loss) * 100 if max_loss > 0 else 0

        return {
            "strategy": "Bear Put Spread",
            "long_put_strike": long_strike,
            "short_put_strike": short_strike,
            "long_put_premium": long_put['price'],
            "short_put_premium": short_put['price'],
            "net_debit": net_debit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven": breakeven,
            "total_delta": total_delta,
            "total_theta": total_theta,
            "probability_of_profit": prob_profit,
            "return_on_capital": roc,
            "days_to_expiration": T * 365
        }

    def run(self):
        """Execute the options spread strategy"""
        symbol = self.options["SYMBOL"]["value"]
        strategy = self.options["STRATEGY"]["value"]
        dte = int(self.options["DAYS_TO_EXPIRATION"]["value"])
        risk_free_rate = float(self.options["RISK_FREE_RATE"]["value"])
        dividend_yield = float(self.options["DIVIDEND_YIELD"]["value"])
        wing_width = float(self.options["WING_WIDTH"]["value"])
        profit_target_pct = float(self.options["PROFIT_TARGET_PCT"]["value"])
        contracts = int(self.options["CONTRACTS"]["value"])

        self.print_status(f"Running Options Spread Strategy: {strategy}")
        self.print_info(f"Symbol: {symbol}")

        # Fetch stock data
        data_fetcher = DataFetcher(self.database)
        df = data_fetcher.get_stock_data(symbol, period="1y", interval="1d")

        if df is None or len(df) < 30:
            self.print_error("Insufficient data for analysis")
            return {"error": "Insufficient data"}

        current_price = df['Close'].iloc[-1]
        self.print_info(f"Current Price: ${current_price:.2f}")

        # Calculate volatility
        hv = calculate_historical_volatility(df['Close'].values, 30)
        sigma = hv * 1.2  # Estimate IV

        self.print_info(f"Estimated IV: {sigma:.2%}")

        # Convert DTE to years
        T = dte / 365.0

        # Execute strategy analysis
        if strategy == "iron_condor":
            analysis = self.iron_condor_analysis(
                current_price, wing_width, T, risk_free_rate, sigma, dividend_yield
            )

        elif strategy == "iron_butterfly":
            analysis = self.iron_butterfly_analysis(
                current_price, wing_width, T, risk_free_rate, sigma, dividend_yield
            )

        elif strategy == "butterfly":
            analysis = self.butterfly_spread_analysis(
                current_price, wing_width, T, risk_free_rate, sigma, "call", dividend_yield
            )

        elif strategy == "calendar":
            T_short = dte / 365.0
            T_long = (dte + 30) / 365.0  # 30 days longer

            analysis = self.calendar_spread_analysis(
                current_price, T_short, T_long, risk_free_rate, sigma, "call", dividend_yield
            )

        elif strategy == "bull_call":
            analysis = self.bull_call_spread_analysis(
                current_price, wing_width, T, risk_free_rate, sigma, dividend_yield
            )

        elif strategy == "bear_put":
            analysis = self.bear_put_spread_analysis(
                current_price, wing_width, T, risk_free_rate, sigma, dividend_yield
            )

        else:
            self.print_error(f"Unknown strategy: {strategy}")
            return {"error": f"Unknown strategy: {strategy}"}

        # Display results
        self.print_good("\n=== Strategy Analysis ===")

        for key, value in analysis.items():
            if isinstance(value, float):
                if 'probability' in key.lower() or 'return' in key.lower():
                    self.print_info(f"{key}: {value:.2f}%")
                else:
                    self.print_info(f"{key}: {value:.4f}")
            else:
                self.print_info(f"{key}: {value}")

        # Calculate position details
        multiplier = 100

        if "net_credit" in analysis:
            credit_per_contract = analysis['net_credit'] * multiplier
            total_credit = credit_per_contract * contracts
            max_loss_total = analysis['max_loss'] * multiplier * contracts

            self.print_info(f"\n=== Position Details (Credit Spread) ===")
            self.print_info(f"Contracts: {contracts}")
            self.print_info(f"Credit per contract: ${credit_per_contract:.2f}")
            self.print_info(f"Total credit received: ${total_credit:.2f}")
            self.print_info(f"Max loss: ${max_loss_total:.2f}")
            self.print_info(f"Capital required: ${max_loss_total:.2f}")

            # Profit target
            profit_target = total_credit * (profit_target_pct / 100)
            self.print_info(f"Profit target ({profit_target_pct}%): ${profit_target:.2f}")

        elif "net_debit" in analysis:
            debit_per_contract = analysis['net_debit'] * multiplier
            total_debit = debit_per_contract * contracts
            max_profit_total = analysis.get('max_profit', 0)
            if isinstance(max_profit_total, (int, float)):
                max_profit_total = max_profit_total * multiplier * contracts

            self.print_info(f"\n=== Position Details (Debit Spread) ===")
            self.print_info(f"Contracts: {contracts}")
            self.print_info(f"Debit per contract: ${debit_per_contract:.2f}")
            self.print_info(f"Total cost: ${total_debit:.2f}")
            if isinstance(max_profit_total, (int, float)):
                self.print_info(f"Max profit: ${max_profit_total:.2f}")

        return {
            "symbol": symbol,
            "current_price": float(current_price),
            "strategy": strategy,
            "analysis": analysis,
            "contracts": contracts
        }
