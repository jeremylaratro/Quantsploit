"""
Options Helpers Module

Provides options chain fetching, Greeks calculation, and strategy suggestions
for the web dashboard.
"""

import sys
from pathlib import Path

# Add parent directory to path so quantsploit module can be imported
_DASHBOARD_DIR = Path(__file__).resolve().parent
_QUANTSPLOIT_ROOT = _DASHBOARD_DIR.parent
if str(_QUANTSPLOIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_QUANTSPLOIT_ROOT))

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, bool):
        return obj
    elif pd.isna(obj):
        return None
    return obj


class MockFramework:
    """Mock framework for standalone execution"""

    def __init__(self):
        self.database = None

    def log(self, message: str, level: str = "info"):
        """Log a message"""
        getattr(logger, level, logger.info)(message)


def get_options_chain(symbol: str, expiration: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch options chain for a symbol with Greeks calculations.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        expiration: Optional expiration date (YYYY-MM-DD)

    Returns:
        Dictionary with calls, puts, expirations, current_price, and greeks
    """
    try:
        from quantsploit.utils.data_fetcher import DataFetcher

        fetcher = DataFetcher(database=None)

        # Get options data
        options_data = fetcher.get_options_chain(symbol, expiration)

        if not options_data:
            return {'success': False, 'error': f'Failed to fetch options for {symbol}'}

        # Get current stock price
        stock_info = fetcher.get_stock_info(symbol)
        current_price = stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 0))

        # Process calls
        calls = options_data.get('calls', pd.DataFrame())
        calls_list = []
        if not calls.empty:
            calls = calls.copy()
            calls['type'] = 'call'
            calls['moneyness'] = calls['strike'].apply(
                lambda x: classify_moneyness(x, current_price, 'call')
            )
            calls['bid_ask_spread'] = calls['ask'] - calls['bid']

            # Add Greeks if we have IV
            if 'impliedVolatility' in calls.columns:
                calls = calculate_greeks_for_chain(calls, current_price, 'call')

            calls_list = calls.to_dict(orient='records')

        # Process puts
        puts = options_data.get('puts', pd.DataFrame())
        puts_list = []
        if not puts.empty:
            puts = puts.copy()
            puts['type'] = 'put'
            puts['moneyness'] = puts['strike'].apply(
                lambda x: classify_moneyness(x, current_price, 'put')
            )
            puts['bid_ask_spread'] = puts['ask'] - puts['bid']

            # Add Greeks if we have IV
            if 'impliedVolatility' in puts.columns:
                puts = calculate_greeks_for_chain(puts, current_price, 'put')

            puts_list = puts.to_dict(orient='records')

        # Calculate Put/Call ratio
        total_call_volume = calls['volume'].sum() if not calls.empty else 0
        total_put_volume = puts['volume'].sum() if not puts.empty else 0
        pcr = total_put_volume / total_call_volume if total_call_volume > 0 else 0

        result = {
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'expiration': options_data.get('expiration'),
            'available_expirations': options_data.get('available_expirations', []),
            'calls': convert_numpy_types(calls_list),
            'puts': convert_numpy_types(puts_list),
            'summary': {
                'total_calls': len(calls_list),
                'total_puts': len(puts_list),
                'call_volume': int(total_call_volume) if total_call_volume else 0,
                'put_volume': int(total_put_volume) if total_put_volume else 0,
                'put_call_ratio': round(pcr, 2),
                'pcr_interpretation': interpret_pcr(pcr)
            }
        }

        return result

    except Exception as e:
        logger.exception(f"Error fetching options chain: {e}")
        return {'success': False, 'error': str(e)}


def classify_moneyness(strike: float, spot: float, option_type: str) -> str:
    """Classify option as ITM, ATM, or OTM"""
    if option_type == 'call':
        if strike < spot * 0.98:
            return "ITM"
        elif strike <= spot * 1.02:
            return "ATM"
        else:
            return "OTM"
    else:  # put
        if strike > spot * 1.02:
            return "ITM"
        elif strike >= spot * 0.98:
            return "ATM"
        else:
            return "OTM"


def interpret_pcr(pcr: float) -> str:
    """Interpret put/call ratio"""
    if pcr > 1.0:
        return "Bearish (more puts)"
    elif pcr < 0.7:
        return "Bullish (more calls)"
    else:
        return "Neutral"


def calculate_greeks_for_chain(df: pd.DataFrame, spot: float, option_type: str,
                               risk_free_rate: float = 0.05) -> pd.DataFrame:
    """
    Calculate Greeks for all options in a chain.

    Uses simplified Black-Scholes approximation for web performance.
    """
    try:
        from scipy.stats import norm

        df = df.copy()

        # Days to expiration (estimate from contract name or default)
        df['dte'] = 30  # Default if not available

        # Convert IV from decimal to percentage
        df['iv'] = df['impliedVolatility'] * 100

        # Calculate time in years
        df['T'] = df['dte'] / 365

        # Calculate d1 for Black-Scholes
        df['d1'] = (np.log(spot / df['strike']) +
                   (risk_free_rate + (df['iv']/100)**2 / 2) * df['T']) / \
                  (df['iv']/100 * np.sqrt(df['T']))

        df['d2'] = df['d1'] - (df['iv']/100 * np.sqrt(df['T']))

        # Delta
        if option_type == 'call':
            df['delta'] = norm.cdf(df['d1'])
        else:
            df['delta'] = norm.cdf(df['d1']) - 1

        # Gamma (same for calls and puts)
        df['gamma'] = norm.pdf(df['d1']) / (spot * df['iv']/100 * np.sqrt(df['T']))

        # Theta (daily)
        df['theta'] = -(spot * norm.pdf(df['d1']) * df['iv']/100) / \
                     (2 * np.sqrt(df['T']) * 365)

        # Vega (per 1% vol change)
        df['vega'] = spot * norm.pdf(df['d1']) * np.sqrt(df['T']) / 100

        # Round the Greeks
        for col in ['delta', 'gamma', 'theta', 'vega']:
            if col in df.columns:
                df[col] = df[col].round(4)

        # Clean up intermediate columns
        df = df.drop(columns=['d1', 'd2', 'T', 'iv'], errors='ignore')

        return df

    except Exception as e:
        logger.warning(f"Error calculating Greeks: {e}")
        return df


def calculate_single_greeks(symbol: str, strike: float, expiration: str,
                            option_type: str, spot: float = None) -> Dict[str, Any]:
    """
    Calculate detailed Greeks for a single option contract.
    """
    try:
        from quantsploit.utils.options_greeks import AdvancedGreeksCalculator, OptionType

        # Get current price if not provided
        if spot is None:
            from quantsploit.utils.data_fetcher import DataFetcher
            fetcher = DataFetcher(database=None)
            stock_info = fetcher.get_stock_info(symbol)
            spot = stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 0))

        # Parse expiration to get time
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        dte = (exp_date - datetime.now()).days
        time_to_expiry = max(dte, 1) / 365

        # Use advanced calculator
        opt_type = OptionType.CALL if option_type.lower() == 'call' else OptionType.PUT

        calculator = AdvancedGreeksCalculator(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=0.05,
            volatility=0.30,  # Default, would need IV from chain
            option_type=opt_type
        )

        greeks = calculator.all_greeks()

        return {
            'success': True,
            'symbol': symbol,
            'strike': strike,
            'expiration': expiration,
            'option_type': option_type,
            'spot': spot,
            'dte': dte,
            'greeks': {
                'delta': round(greeks.delta, 4),
                'gamma': round(greeks.gamma, 4),
                'theta': round(greeks.theta, 4),
                'vega': round(greeks.vega, 4),
                'rho': round(greeks.rho, 4),
                'vanna': round(greeks.vanna, 4),
                'volga': round(greeks.volga, 4),
                'charm': round(greeks.charm, 4),
                'price': round(greeks.price, 4)
            }
        }

    except Exception as e:
        logger.exception(f"Error calculating Greeks: {e}")
        return {'success': False, 'error': str(e)}


def suggest_strategies(symbol: str, current_price: float, iv_rank: float = 50,
                       outlook: str = 'neutral') -> List[Dict[str, Any]]:
    """
    Suggest options strategies based on market conditions.

    Args:
        symbol: Stock symbol
        current_price: Current stock price
        iv_rank: IV rank (0-100)
        outlook: Market outlook (bullish, bearish, neutral)

    Returns:
        List of suggested strategies with descriptions
    """
    strategies = []

    # High IV - prefer selling strategies
    if iv_rank > 70:
        if outlook == 'bullish':
            strategies.append({
                'name': 'Cash-Secured Put',
                'description': 'Sell OTM puts to collect premium while waiting to buy shares',
                'max_profit': 'Limited to premium received',
                'max_loss': 'Strike price - premium (if assigned)',
                'best_for': 'High IV, bullish outlook'
            })
            strategies.append({
                'name': 'Bull Put Spread',
                'description': 'Sell higher strike put, buy lower strike put',
                'max_profit': 'Net credit received',
                'max_loss': 'Width of spread - credit',
                'best_for': 'High IV, moderately bullish'
            })
        elif outlook == 'bearish':
            strategies.append({
                'name': 'Covered Call',
                'description': 'Own shares and sell OTM calls',
                'max_profit': 'Premium + (strike - cost basis)',
                'max_loss': 'Cost basis - premium',
                'best_for': 'High IV, neutral to bearish'
            })
            strategies.append({
                'name': 'Bear Call Spread',
                'description': 'Sell lower strike call, buy higher strike call',
                'max_profit': 'Net credit received',
                'max_loss': 'Width of spread - credit',
                'best_for': 'High IV, moderately bearish'
            })
        else:  # neutral
            strategies.append({
                'name': 'Iron Condor',
                'description': 'Sell OTM call spread + sell OTM put spread',
                'max_profit': 'Net credit from both spreads',
                'max_loss': 'Width of wider spread - total credit',
                'best_for': 'High IV, expecting range-bound'
            })
            strategies.append({
                'name': 'Short Straddle',
                'description': 'Sell ATM call + sell ATM put',
                'max_profit': 'Total premium received',
                'max_loss': 'Unlimited',
                'best_for': 'High IV, expecting low volatility'
            })

    # Low IV - prefer buying strategies
    elif iv_rank < 30:
        if outlook == 'bullish':
            strategies.append({
                'name': 'Long Call',
                'description': 'Buy calls for leveraged upside exposure',
                'max_profit': 'Unlimited',
                'max_loss': 'Premium paid',
                'best_for': 'Low IV, strongly bullish'
            })
            strategies.append({
                'name': 'Call Debit Spread',
                'description': 'Buy lower strike call, sell higher strike call',
                'max_profit': 'Width of spread - debit paid',
                'max_loss': 'Net debit paid',
                'best_for': 'Low IV, moderately bullish'
            })
        elif outlook == 'bearish':
            strategies.append({
                'name': 'Long Put',
                'description': 'Buy puts for downside protection or speculation',
                'max_profit': 'Strike price - premium',
                'max_loss': 'Premium paid',
                'best_for': 'Low IV, strongly bearish'
            })
            strategies.append({
                'name': 'Put Debit Spread',
                'description': 'Buy higher strike put, sell lower strike put',
                'max_profit': 'Width of spread - debit paid',
                'max_loss': 'Net debit paid',
                'best_for': 'Low IV, moderately bearish'
            })
        else:  # neutral
            strategies.append({
                'name': 'Long Straddle',
                'description': 'Buy ATM call + buy ATM put',
                'max_profit': 'Unlimited',
                'max_loss': 'Total premium paid',
                'best_for': 'Low IV, expecting big move'
            })
            strategies.append({
                'name': 'Calendar Spread',
                'description': 'Sell near-term option, buy longer-term option',
                'max_profit': 'Complex (depends on IV changes)',
                'max_loss': 'Net debit paid',
                'best_for': 'Low IV expected to rise'
            })

    # Medium IV - mixed strategies
    else:
        strategies.append({
            'name': 'Vertical Spread',
            'description': 'Bull or bear spread based on outlook',
            'max_profit': 'Defined by spread width',
            'max_loss': 'Defined by spread width',
            'best_for': 'Medium IV, directional view'
        })
        strategies.append({
            'name': 'Diagonal Spread',
            'description': 'Different strikes AND expirations',
            'max_profit': 'Complex',
            'max_loss': 'Net debit/credit',
            'best_for': 'Medium IV, time and direction play'
        })

    return strategies
