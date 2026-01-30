"""
Strategy API Helper Module

Provides introspection and execution capabilities for all trading strategies
via the web dashboard.
"""

import sys
from pathlib import Path

# Add parent directory to path so quantsploit module can be imported
_DASHBOARD_DIR = Path(__file__).resolve().parent
_QUANTSPLOIT_ROOT = _DASHBOARD_DIR.parent
if str(_QUANTSPLOIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_QUANTSPLOIT_ROOT))

import importlib
import inspect
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Strategy module mappings for BaseModule strategies (framework-compatible)
STRATEGY_MODULES = {
    # Core strategies
    'sma_crossover': ('quantsploit.modules.strategies.sma_crossover', 'SMACrossover'),
    'mean_reversion': ('quantsploit.modules.strategies.mean_reversion', 'MeanReversion'),
    'momentum_signals': ('quantsploit.modules.strategies.momentum_signals', 'MomentumSignals'),
    'multifactor_scoring': ('quantsploit.modules.strategies.multifactor_scoring', 'MultiFactorScoring'),
    # Advanced strategies
    'kalman_adaptive': ('quantsploit.modules.strategies.kalman_adaptive', 'KalmanAdaptiveStrategy'),
    'volume_profile_swing': ('quantsploit.modules.strategies.volume_profile_swing', 'VolumeProfileSwingStrategy'),
    'hmm_regime_detection': ('quantsploit.modules.strategies.hmm_regime_detection', 'HMMRegimeDetectionStrategy'),
    'ml_swing_trading': ('quantsploit.modules.strategies.ml_swing_trading', 'MLSwingTradingStrategy'),
    'pairs_trading': ('quantsploit.modules.strategies.pairs_trading', 'PairsTradingStrategy'),
    'options_volatility': ('quantsploit.modules.strategies.options_volatility', 'OptionsVolatilityStrategy'),
    'options_spreads': ('quantsploit.modules.strategies.options_spreads', 'OptionsSpreadStrategy'),
    'reddit_sentiment_strategy': ('quantsploit.modules.strategies.reddit_sentiment_strategy', 'RedditSentimentStrategy'),
}

# v0.2.0 strategies - these are standalone analysis classes, not BaseModule subclasses
# They have different interfaces and require data to be passed in, not fetched via options
V020_STRATEGY_METADATA = {
    'risk_parity': {
        'name': 'Risk Parity Strategy',
        'description': 'Equalizes risk contribution across assets using inverse volatility or ERC optimization',
        'category': 'v0.2.0',
        'module': ('quantsploit.modules.strategies.risk_parity', 'RiskParityStrategy'),
        'options': {
            'SYMBOLS': {'value': 'AAPL,MSFT,GOOGL,AMZN', 'required': True, 'description': 'Comma-separated list of symbols'},
            'PERIOD': {'value': '1y', 'required': False, 'description': 'Historical data period'},
            'METHOD': {'value': 'inverse_volatility', 'required': False, 'description': 'Risk parity method: inverse_volatility, erc, hrp'},
            'TARGET_VOL': {'value': 0.10, 'required': False, 'description': 'Target portfolio volatility (annualized)'},
        },
        'trading_guide': """SYNOPSIS: Allocates portfolio weights to equalize risk contribution across assets.

SIMULATION POSITIONS:
  - Calculates optimal weights based on inverse volatility or equal risk contribution
  - Rebalances periodically to maintain risk parity
  - Can use leverage to target specific volatility levels

RECOMMENDED USAGE:
  - Use with diverse asset classes (stocks, bonds, commodities)
  - Best for long-term portfolio construction
  - Methods: inverse_volatility (simple), erc (optimal), hrp (hierarchical clustering)"""
    },
    'volatility_breakout': {
        'name': 'Volatility Breakout Strategy',
        'description': 'Trades breakouts when price moves beyond volatility-based channels',
        'category': 'v0.2.0',
        'module': ('quantsploit.modules.strategies.volatility_breakout', 'VolatilityBreakoutStrategy'),
        'options': {
            'SYMBOL': {'value': 'SPY', 'required': True, 'description': 'Stock symbol'},
            'PERIOD': {'value': '1y', 'required': False, 'description': 'Historical data period'},
            'ATR_MULTIPLIER': {'value': 2.0, 'required': False, 'description': 'ATR multiplier for channel width'},
            'LOOKBACK': {'value': 20, 'required': False, 'description': 'Lookback period for volatility calculation'},
        },
        'trading_guide': """SYNOPSIS: Enters positions when price breaks volatility bands.

SIMULATION POSITIONS:
  - LONG: When price breaks above upper volatility band
  - SHORT: When price breaks below lower volatility band
  - Exits when price returns to mean or hits stop

RECOMMENDED USAGE:
  - Best in trending markets with expanding volatility
  - Use tighter ATR multiplier (1.5-2.0) for more signals
  - Combine with volume confirmation for higher quality signals"""
    },
    'fama_french': {
        'name': 'Fama-French Factor Strategy',
        'description': 'Multi-factor model using market, size, value, momentum, and quality factors',
        'category': 'v0.2.0',
        'module': ('quantsploit.modules.strategies.fama_french', 'FamaFrenchStrategy'),
        'options': {
            'SYMBOLS': {'value': 'SP500', 'required': True, 'description': 'Symbol list or index name'},
            'PERIOD': {'value': '2y', 'required': False, 'description': 'Historical data period'},
            'FACTORS': {'value': 'market,size,value,momentum', 'required': False, 'description': 'Factors to use'},
            'TOP_N': {'value': 20, 'required': False, 'description': 'Number of top stocks to select'},
        },
        'trading_guide': """SYNOPSIS: Ranks and selects stocks based on multi-factor scores.

FACTORS:
  - Market: Beta exposure to market returns
  - Size (SMB): Small minus big market cap
  - Value (HML): High minus low book-to-market
  - Momentum: Recent price performance
  - Quality: Profitability and investment patterns

RECOMMENDED USAGE:
  - Rebalance monthly or quarterly
  - Combine multiple factors for diversification
  - Use for long-only equity portfolios"""
    },
    'earnings_momentum': {
        'name': 'Earnings Momentum Strategy',
        'description': 'Trades post-earnings announcement drift based on earnings surprises',
        'category': 'v0.2.0',
        'module': ('quantsploit.modules.strategies.earnings_momentum', 'EarningsMomentumStrategy'),
        'options': {
            'SYMBOLS': {'value': 'AAPL,MSFT,GOOGL', 'required': True, 'description': 'Symbols to analyze'},
            'PERIOD': {'value': '1y', 'required': False, 'description': 'Historical data period'},
            'SURPRISE_THRESHOLD': {'value': 5.0, 'required': False, 'description': 'Minimum earnings surprise %'},
            'HOLDING_DAYS': {'value': 60, 'required': False, 'description': 'Days to hold after earnings'},
        },
        'trading_guide': """SYNOPSIS: Exploits post-earnings announcement drift (PEAD) anomaly.

SIMULATION POSITIONS:
  - LONG: Positive earnings surprise > threshold
  - SHORT: Negative earnings surprise < -threshold
  - Holds for 60-90 days to capture drift

RECOMMENDED USAGE:
  - Focus on small/mid-cap stocks with lower analyst coverage
  - Combine with revision momentum for stronger signals
  - Best to enter 1-2 days after earnings announcement"""
    },
    'adaptive_allocation': {
        'name': 'Adaptive Asset Allocation',
        'description': 'Dynamically adjusts allocation based on market regime detection',
        'category': 'v0.2.0',
        'module': ('quantsploit.modules.strategies.adaptive_allocation', 'AdaptiveAssetAllocation'),
        'options': {
            'SYMBOLS': {'value': 'SPY,TLT,GLD,VNQ', 'required': True, 'description': 'Asset symbols'},
            'PERIOD': {'value': '2y', 'required': False, 'description': 'Historical data period'},
            'REGIME_LOOKBACK': {'value': 60, 'required': False, 'description': 'Days for regime detection'},
            'REBALANCE_FREQ': {'value': 'monthly', 'required': False, 'description': 'Rebalancing frequency'},
        },
        'trading_guide': """SYNOPSIS: Adjusts asset allocation based on detected market regime.

REGIMES:
  - BULL: Risk-on, higher equity allocation
  - BEAR: Risk-off, higher bond/gold allocation
  - SIDEWAYS: Balanced allocation
  - CRISIS: Maximum defensive positioning

RECOMMENDED USAGE:
  - Use diverse asset classes (stocks, bonds, commodities, real estate)
  - Monthly rebalancing is typically sufficient
  - Can layer with momentum for asset selection within classes"""
    },
    'options_vol_arb': {
        'name': 'Options Volatility Arbitrage',
        'description': 'Trades implied vs realized volatility discrepancies',
        'category': 'v0.2.0',
        'module': ('quantsploit.modules.strategies.options_vol_arb', 'OptionsVolatilityArbitrage'),
        'options': {
            'SYMBOL': {'value': 'SPY', 'required': True, 'description': 'Underlying symbol'},
            'PERIOD': {'value': '1y', 'required': False, 'description': 'Historical data period'},
            'VOL_THRESHOLD': {'value': 5.0, 'required': False, 'description': 'IV-RV spread threshold %'},
            'STRATEGY': {'value': 'straddle', 'required': False, 'description': 'Options strategy type'},
        },
        'trading_guide': """SYNOPSIS: Profits from mean reversion in implied vs realized volatility.

STRATEGIES:
  - SELL VOL: When IV >> RV, sell straddles/strangles
  - BUY VOL: When IV << RV, buy straddles/strangles
  - VARIANCE SWAP: Pure vol exposure without directional risk

RECOMMENDED USAGE:
  - Most effective on liquid underlyings (SPY, QQQ, major indices)
  - Requires options approval and understanding of Greeks
  - Monitor Vega exposure carefully"""
    },
    'vwap_execution': {
        'name': 'VWAP Execution Strategy',
        'description': 'Executes large orders tracking Volume-Weighted Average Price',
        'category': 'v0.2.0',
        'module': ('quantsploit.modules.strategies.vwap_execution', 'VWAPExecutionStrategy'),
        'options': {
            'SYMBOL': {'value': 'AAPL', 'required': True, 'description': 'Symbol to execute'},
            'SHARES': {'value': 10000, 'required': True, 'description': 'Total shares to execute'},
            'START_TIME': {'value': '09:30', 'required': False, 'description': 'Execution start time'},
            'END_TIME': {'value': '16:00', 'required': False, 'description': 'Execution end time'},
            'URGENCY': {'value': 'medium', 'required': False, 'description': 'Execution urgency: low, medium, high'},
        },
        'trading_guide': """SYNOPSIS: Minimizes market impact by slicing orders across the day.

ALGORITHM:
  - Predicts intraday volume profile
  - Slices order proportional to expected volume
  - Targets VWAP benchmark for execution quality

RECOMMENDED USAGE:
  - Use for large institutional orders (>1% of ADV)
  - Lower urgency = better VWAP tracking
  - Higher urgency = faster completion but more impact"""
    },
}


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
    """Mock framework for standalone strategy execution"""

    def __init__(self):
        self.database = None

    def log(self, message: str, level: str = "info"):
        """Log a message"""
        getattr(logger, level, logger.info)(message)


def get_strategy_class(strategy_id: str):
    """
    Dynamically import and return a strategy class.

    Args:
        strategy_id: Strategy identifier (e.g., 'sma_crossover')

    Returns:
        Strategy class or None if not found
    """
    if strategy_id not in STRATEGY_MODULES:
        return None

    module_path, class_name = STRATEGY_MODULES[strategy_id]

    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name, None)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import strategy {strategy_id}: {e}")
        return None


def get_all_strategies() -> List[Dict[str, Any]]:
    """
    Get all available strategies with their metadata and options.

    Returns:
        List of strategy dictionaries with id, name, description, category, and options
    """
    strategies = []
    framework = MockFramework()

    # Load BaseModule strategies (core and advanced)
    for strategy_id, (module_path, class_name) in STRATEGY_MODULES.items():
        try:
            strategy_class = get_strategy_class(strategy_id)
            if strategy_class is None:
                continue

            # Instantiate to get options
            instance = strategy_class(framework)

            # Determine category
            category = 'core'
            if strategy_id in ['kalman_adaptive', 'volume_profile_swing', 'hmm_regime_detection',
                              'ml_swing_trading', 'pairs_trading', 'options_volatility',
                              'options_spreads', 'reddit_sentiment_strategy']:
                category = 'advanced'

            # Get trading guide if available
            trading_guide = None
            if hasattr(instance, 'trading_guide'):
                trading_guide = instance.trading_guide()

            strategies.append({
                'id': strategy_id,
                'name': instance.name,
                'description': instance.description,
                'category': category,
                'options': instance.options,
                'trading_guide': trading_guide
            })

        except Exception as e:
            logger.warning(f"Failed to load strategy {strategy_id}: {e}")
            continue

    # Add v0.2.0 strategies from metadata (these don't inherit from BaseModule)
    for strategy_id, metadata in V020_STRATEGY_METADATA.items():
        strategies.append({
            'id': strategy_id,
            'name': metadata['name'],
            'description': metadata['description'],
            'category': metadata['category'],
            'options': metadata['options'],
            'trading_guide': metadata.get('trading_guide')
        })

    return strategies


def get_strategy_options(strategy_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the options schema for a specific strategy.

    Args:
        strategy_id: Strategy identifier

    Returns:
        Dictionary of options or None if strategy not found
    """
    # Check v0.2.0 strategies first
    if strategy_id in V020_STRATEGY_METADATA:
        metadata = V020_STRATEGY_METADATA[strategy_id]
        return {
            'strategy_id': strategy_id,
            'name': metadata['name'],
            'description': metadata['description'],
            'options': metadata['options'],
            'trading_guide': metadata.get('trading_guide')
        }

    # Try BaseModule strategies
    strategy_class = get_strategy_class(strategy_id)
    if strategy_class is None:
        return None

    framework = MockFramework()
    instance = strategy_class(framework)

    return {
        'strategy_id': strategy_id,
        'name': instance.name,
        'description': instance.description,
        'options': instance.options,
        'trading_guide': instance.trading_guide() if hasattr(instance, 'trading_guide') else None
    }


def execute_strategy(strategy_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a strategy with the given options.

    Args:
        strategy_id: Strategy identifier
        options: Dictionary of option key-value pairs

    Returns:
        Dictionary with execution results
    """
    # Handle v0.2.0 strategies separately
    if strategy_id in V020_STRATEGY_METADATA:
        return execute_v020_strategy(strategy_id, options)

    # Handle BaseModule strategies
    strategy_class = get_strategy_class(strategy_id)
    if strategy_class is None:
        return {'success': False, 'error': f'Strategy {strategy_id} not found'}

    framework = MockFramework()

    try:
        instance = strategy_class(framework)

        # Set options
        for key, value in options.items():
            if not instance.set_option(key, value):
                logger.warning(f"Unknown option {key} for strategy {strategy_id}")

        # Validate
        valid, message = instance.validate_options()
        if not valid:
            return {'success': False, 'error': message}

        # Execute
        results = instance.run()

        # Convert to JSON-safe types
        results = convert_numpy_types(results)

        return {
            'success': True,
            'strategy_id': strategy_id,
            'strategy_name': instance.name,
            'results': results
        }

    except Exception as e:
        logger.exception(f"Strategy execution failed: {e}")
        return {'success': False, 'error': str(e)}


def execute_v020_strategy(strategy_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a v0.2.0 strategy that requires data-driven initialization.

    These strategies don't inherit from BaseModule and have different interfaces.
    They require fetching data first and passing it to the strategy.
    """
    from quantsploit.utils.data_fetcher import DataFetcher

    metadata = V020_STRATEGY_METADATA.get(strategy_id)
    if not metadata:
        return {'success': False, 'error': f'v0.2.0 strategy {strategy_id} not found'}

    try:
        fetcher = DataFetcher(database=None)

        # Import the strategy class
        module_path, class_name = metadata['module']
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)

        # Get parameters from options
        period = options.get('PERIOD', '1y')

        # Different strategies have different data requirements
        if strategy_id == 'risk_parity':
            symbols_str = options.get('SYMBOLS', 'AAPL,MSFT,GOOGL,AMZN')
            symbols = [s.strip() for s in symbols_str.split(',')]

            # Fetch returns data for all symbols
            returns_data = {}
            for symbol in symbols:
                df = fetcher.get_stock_data(symbol, period, '1d')
                if df is not None and not df.empty:
                    returns_data[symbol] = df['Close'].pct_change().dropna()

            if len(returns_data) < 2:
                return {'success': False, 'error': 'Need at least 2 symbols with valid data'}

            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data).dropna()

            # Instantiate and run strategy
            strategy = strategy_class(returns_df)
            method = options.get('METHOD', 'inverse_volatility')

            if method == 'inverse_volatility':
                result = strategy.inverse_volatility_weights()
            elif method == 'erc':
                result = strategy.equal_risk_contribution()
            elif method == 'hrp':
                result = strategy.hierarchical_risk_parity()
            else:
                result = strategy.inverse_volatility_weights()

            return {
                'success': True,
                'strategy_id': strategy_id,
                'strategy_name': metadata['name'],
                'results': {
                    'weights': convert_numpy_types(result.weights.tolist()),
                    'risk_contributions': convert_numpy_types(result.risk_contributions.tolist()),
                    'portfolio_volatility': float(result.portfolio_volatility),
                    'effective_n_assets': float(result.effective_n_assets),
                    'method': result.method,
                    'symbols': symbols,
                    'summary': result.summary()
                }
            }

        elif strategy_id == 'volatility_breakout':
            symbol = options.get('SYMBOL', 'SPY')
            df = fetcher.get_stock_data(symbol, period, '1d')

            if df is None or df.empty:
                return {'success': False, 'error': f'No data found for {symbol}'}

            strategy = strategy_class(
                df,
                bb_period=int(options.get('BB_PERIOD', 20)),
                bb_std=float(options.get('BB_STD', 2.0)),
                kc_period=int(options.get('KC_PERIOD', 20)),
                kc_mult=float(options.get('KC_MULT', 1.5))
            )
            signals = strategy.generate_signals(
                min_squeeze_duration=int(options.get('MIN_SQUEEZE_DURATION', 5))
            )

            return {
                'success': True,
                'strategy_id': strategy_id,
                'strategy_name': metadata['name'],
                'results': {
                    'symbol': symbol,
                    'signals': convert_numpy_types([{
                        'date': str(s.date),
                        'direction': s.direction,
                        'strength': s.strength,
                        'squeeze_duration': s.squeeze_duration,
                        'atr_expansion': s.atr_expansion,
                        'volume_confirmation': s.volume_confirmation
                    } for s in signals[-10:]]),  # Last 10 signals
                    'total_signals': len(signals),
                    'current_in_squeeze': bool(strategy.detect_squeeze().iloc[-1]) if len(strategy.data) > 0 else False
                }
            }

        elif strategy_id == 'fama_french':
            # Fama-French factor analysis - works on single symbol with momentum/quality proxies
            symbol = options.get('SYMBOL') or options.get('SYMBOLS', 'AAPL').split(',')[0]
            df = fetcher.get_stock_data(symbol, period, '1d')

            if df is None or df.empty:
                return {'success': False, 'error': f'No data found for {symbol}'}

            # Calculate factor proxies
            close = df['Close']
            returns = close.pct_change().dropna()

            # Momentum factor (60-day ROC)
            lookback = int(options.get('LOOKBACK', 60))
            if len(close) > lookback:
                momentum = (close.iloc[-1] - close.iloc[-lookback]) / close.iloc[-lookback]
            else:
                momentum = 0

            # Volatility proxy for quality
            vol_20 = returns.iloc[-20:].std() if len(returns) >= 20 else returns.std()
            vol_60 = returns.iloc[-60:].std() if len(returns) >= 60 else returns.std()
            quality_score = 1 - (vol_20 / vol_60) if vol_60 > 0 else 0.5

            # Composite score
            factor_score = (momentum * 50 + quality_score * 50)

            return {
                'success': True,
                'strategy_id': strategy_id,
                'strategy_name': metadata['name'],
                'results': {
                    'symbol': symbol,
                    'factor_scores': {
                        'momentum': float(momentum),
                        'quality_proxy': float(quality_score),
                        'composite': float(factor_score)
                    },
                    'signal': 'LONG' if factor_score > 0.6 else ('SHORT' if factor_score < 0.4 else 'NEUTRAL'),
                    'note': 'Using momentum and volatility as factor proxies. Full factor analysis requires multi-stock universe.'
                }
            }

        elif strategy_id == 'adaptive_allocation':
            # Adaptive allocation - regime detection for single symbol
            symbols_str = options.get('SYMBOLS', 'SPY,TLT,GLD')
            symbols = [s.strip() for s in symbols_str.split(',')]

            # Fetch data for primary symbol
            symbol = symbols[0]
            df = fetcher.get_stock_data(symbol, period, '1d')

            if df is None or df.empty:
                return {'success': False, 'error': f'No data found for {symbol}'}

            close = df['Close']
            returns = close.pct_change().dropna()

            # Regime detection
            momentum_lookback = int(options.get('REGIME_LOOKBACK', 60))
            vol_lookback = 20

            roc = (close.iloc[-1] - close.iloc[-momentum_lookback]) / close.iloc[-momentum_lookback] if len(close) > momentum_lookback else 0
            current_vol = returns.iloc[-vol_lookback:].std() * np.sqrt(252) if len(returns) >= vol_lookback else 0.2
            historical_vol = returns.iloc[-momentum_lookback:].std() * np.sqrt(252) if len(returns) >= momentum_lookback else 0.2
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.iloc[-1]
            trend_bullish = close.iloc[-1] > sma_50

            # Determine regime
            if roc > 0.05 and trend_bullish:
                regime = 'BULL'
                allocation_suggestion = 'Risk-on: Higher equity allocation'
            elif roc < -0.05 or not trend_bullish:
                regime = 'BEAR'
                allocation_suggestion = 'Risk-off: Higher bond/gold allocation'
            else:
                regime = 'SIDEWAYS'
                allocation_suggestion = 'Neutral: Balanced allocation'

            vol_elevated = current_vol > historical_vol * 1.3

            return {
                'success': True,
                'strategy_id': strategy_id,
                'strategy_name': metadata['name'],
                'results': {
                    'symbol': symbol,
                    'regime': regime,
                    'regime_indicators': {
                        'momentum_roc': float(roc),
                        'current_volatility': float(current_vol),
                        'historical_volatility': float(historical_vol),
                        'volatility_elevated': vol_elevated,
                        'price_above_sma50': bool(trend_bullish)
                    },
                    'allocation_suggestion': allocation_suggestion
                }
            }

        elif strategy_id == 'earnings_momentum':
            return {
                'success': True,
                'strategy_id': strategy_id,
                'strategy_name': metadata['name'],
                'results': {
                    'status': 'requires_data',
                    'message': 'Earnings Momentum requires earnings announcement data (EPS, estimates, dates) which must be configured from an external provider.',
                    'required_data': ['earnings_dates', 'actual_eps', 'estimated_eps'],
                    'trading_guide': metadata.get('trading_guide', '')
                }
            }

        elif strategy_id == 'options_vol_arb':
            return {
                'success': True,
                'strategy_id': strategy_id,
                'strategy_name': metadata['name'],
                'results': {
                    'status': 'requires_data',
                    'message': 'Options Vol Arb requires implied volatility data from options chains which must be configured from an external provider.',
                    'required_data': ['implied_volatility', 'options_chain', 'greeks'],
                    'trading_guide': metadata.get('trading_guide', '')
                }
            }

        elif strategy_id == 'vwap_execution':
            return {
                'success': True,
                'strategy_id': strategy_id,
                'strategy_name': metadata['name'],
                'results': {
                    'status': 'requires_data',
                    'message': 'VWAP Execution requires intraday (minute-level) price and volume data. This is an execution algorithm for order slicing, not a signal generator.',
                    'required_data': ['intraday_prices', 'intraday_volume', 'volume_profile'],
                    'trading_guide': metadata.get('trading_guide', '')
                }
            }

        else:
            return {'success': False, 'error': f'Execution not implemented for {strategy_id}'}

    except Exception as e:
        logger.exception(f"v0.2.0 strategy execution failed: {e}")
        return {'success': False, 'error': str(e)}


def get_strategy_categories() -> Dict[str, List[str]]:
    """
    Get strategies organized by category.

    Returns:
        Dictionary mapping category names to lists of strategy IDs
    """
    categories = {
        'core': [],
        'advanced': [],
        'v0.2.0': []
    }

    core = ['sma_crossover', 'mean_reversion', 'momentum_signals', 'multifactor_scoring']
    advanced = ['kalman_adaptive', 'volume_profile_swing', 'hmm_regime_detection',
                'ml_swing_trading', 'pairs_trading', 'options_volatility',
                'options_spreads', 'reddit_sentiment_strategy']
    v020 = ['risk_parity', 'volatility_breakout', 'fama_french', 'earnings_momentum',
            'adaptive_allocation', 'options_vol_arb', 'vwap_execution']

    categories['core'] = core
    categories['advanced'] = advanced
    categories['v0.2.0'] = v020

    return categories
