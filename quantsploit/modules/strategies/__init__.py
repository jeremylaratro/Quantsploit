"""
Trading strategy modules

This package contains various trading strategies for backtesting and analysis.
"""

# Core strategies
from .sma_crossover import SMACrossover
from .mean_reversion import MeanReversion
from .momentum_signals import MomentumSignals

# Advanced strategies
from .pairs_trading import PairsTradingStrategy
from .options_volatility import OptionsVolatilityStrategy
from .options_spreads import OptionsSpreadStrategy
from .multifactor_scoring import MultiFactorScoring
from .volume_profile_swing import VolumeProfileSwingStrategy
from .ml_swing_trading import MLSwingTradingStrategy
from .hmm_regime_detection import HMMRegimeDetectionStrategy
from .kalman_adaptive import KalmanAdaptiveStrategy
from .reddit_sentiment_strategy import RedditSentimentStrategy

# New strategies (v0.2.0)
from .risk_parity import RiskParityStrategy, RiskParityResult
from .volatility_breakout import VolatilityBreakoutStrategy, BreakoutSignal
from .fama_french import FamaFrenchStrategy, FactorExposure
from .earnings_momentum import EarningsMomentumStrategy, EarningsSurprise, EarningsMomentumSignal
from .adaptive_allocation import AdaptiveAssetAllocation, ProtectiveAssetAllocation, MarketRegime
from .options_vol_arb import OptionsVolatilityArbitrage, VarianceSwapReplicator, VolatilitySignal
from .vwap_execution import VWAPExecutionStrategy, VolumeProfile, DynamicVWAPExecutor, ExecutionReport

__all__ = [
    # Core
    'SMACrossover',
    'MeanReversion',
    'MomentumSignals',
    # Advanced
    'PairsTradingStrategy',
    'OptionsVolatilityStrategy',
    'OptionsSpreadStrategy',
    'MultiFactorScoring',
    'VolumeProfileSwingStrategy',
    'MLSwingTradingStrategy',
    'HMMRegimeDetectionStrategy',
    'KalmanAdaptiveStrategy',
    'RedditSentimentStrategy',
    # New (v0.2.0)
    'RiskParityStrategy',
    'RiskParityResult',
    'VolatilityBreakoutStrategy',
    'BreakoutSignal',
    'FamaFrenchStrategy',
    'FactorExposure',
    'EarningsMomentumStrategy',
    'EarningsSurprise',
    'EarningsMomentumSignal',
    'AdaptiveAssetAllocation',
    'ProtectiveAssetAllocation',
    'MarketRegime',
    'OptionsVolatilityArbitrage',
    'VarianceSwapReplicator',
    'VolatilitySignal',
    'VWAPExecutionStrategy',
    'VolumeProfile',
    'DynamicVWAPExecutor',
    'ExecutionReport',
]
