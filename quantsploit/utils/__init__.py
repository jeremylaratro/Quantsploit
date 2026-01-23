"""
Utility modules for Quantsploit quantitative trading framework.

This module provides access to all utility functions and classes including:
- Data fetching and validation
- Backtesting and walk-forward optimization
- Monte Carlo simulation
- Position sizing (Kelly Criterion, volatility-adjusted)
- Transaction cost modeling
- Portfolio optimization (Markowitz, Risk Parity)
- Volatility modeling (GARCH, regime detection)
- Stress testing framework
- Structured logging
"""

from .data_fetcher import DataFetcher
from .helpers import format_currency, format_percentage, format_table
from .ticker_validator import TickerValidator, get_validator

# Logging configuration
from .logging_config import (
    QuantsploitLogger,
    LogConfig,
    LogLevel,
    EventType,
    PerformanceTracker,
    setup_logging,
    get_logger,
    log_function_call,
)

# Walk-forward optimization
from .walk_forward import (
    WalkForwardOptimizer,
    WalkForwardMode,
    WalkForwardWindow,
    WalkForwardResult,
    WalkForwardReport,
    calculate_walk_forward_efficiency,
    run_quick_walk_forward,
)

# Monte Carlo simulation
from .monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResults,
    SimulationConfig,
    SimulationResult,
    ConfidenceInterval,
    DistributionReport,
    RandomizationMethod,
    run_monte_carlo_analysis,
)

# Position sizing
from .position_sizing import (
    KellyCriterion,
    VolatilityAdjustedSizing,
    PositionSizer,
    PositionSizeResult,
    SizingMethod,
    calculate_kelly,
    calculate_half_kelly,
    calculate_atr_shares,
)

# Transaction cost modeling
from .transaction_costs import (
    TransactionCostModel,
    CostAwareBacktester,
    CostAwareBacktestConfig,
    CostAwareBacktestResults,
    TransactionCostBreakdown,
    CostProfile,
    LiquidityTier,
    CommissionTier,
    MarketImpactParams,
    create_cost_model,
    estimate_transaction_costs,
)

# Portfolio optimization
from .portfolio_optimizer import (
    MarkowitzOptimizer,
    PortfolioConstraints,
    PortfolioMetrics,
    OptimizationObjective,
    optimize_portfolio,
    create_sample_returns,
)

# Data validation
from .data_validation import (
    DataValidator,
    DataCleaner,
    MissingDataHandler,
    ValidationIssue,
    ValidationSeverity,
    MissingDataStrategy,
    QualityReport,
    validate_ohlcv_data,
    clean_ohlcv_data,
    get_data_quality_score,
)

# Volatility modeling
from .volatility_models import (
    GARCHModel,
    GARCHFitResult,
    VolatilityForecast,
    VolatilityRegime,
    VolatilityRegimeDetector,
    GARCHBacktestIntegration,
    calculate_dynamic_var,
    calculate_dynamic_stop_loss,
    calculate_garch_adjusted_position_size,
    ARCH_AVAILABLE,
)

# Stress testing
from .stress_testing import (
    StressTestFramework,
    ScenarioGenerator,
    StressTestResult,
    StressTestReport,
    HistoricalScenario,
    HypotheticalScenario,
    ReverseStressResult,
    run_full_stress_test,
)

__all__ = [
    # Core utilities
    'DataFetcher',
    'format_currency',
    'format_percentage',
    'format_table',
    'TickerValidator',
    'get_validator',

    # Logging
    'QuantsploitLogger',
    'LogConfig',
    'LogLevel',
    'EventType',
    'PerformanceTracker',
    'setup_logging',
    'get_logger',
    'log_function_call',

    # Walk-forward optimization
    'WalkForwardOptimizer',
    'WalkForwardMode',
    'WalkForwardWindow',
    'WalkForwardResult',
    'WalkForwardReport',
    'calculate_walk_forward_efficiency',
    'run_quick_walk_forward',

    # Monte Carlo simulation
    'MonteCarloSimulator',
    'MonteCarloResults',
    'SimulationConfig',
    'SimulationResult',
    'ConfidenceInterval',
    'DistributionReport',
    'RandomizationMethod',
    'run_monte_carlo_analysis',

    # Position sizing
    'KellyCriterion',
    'VolatilityAdjustedSizing',
    'PositionSizer',
    'PositionSizeResult',
    'SizingMethod',
    'calculate_kelly',
    'calculate_half_kelly',
    'calculate_atr_shares',

    # Transaction cost modeling
    'TransactionCostModel',
    'CostAwareBacktester',
    'CostAwareBacktestConfig',
    'CostAwareBacktestResults',
    'TransactionCostBreakdown',
    'CostProfile',
    'LiquidityTier',
    'CommissionTier',
    'MarketImpactParams',
    'create_cost_model',
    'estimate_transaction_costs',

    # Portfolio optimization
    'MarkowitzOptimizer',
    'PortfolioConstraints',
    'PortfolioMetrics',
    'OptimizationObjective',
    'optimize_portfolio',
    'create_sample_returns',

    # Data validation
    'DataValidator',
    'DataCleaner',
    'MissingDataHandler',
    'ValidationIssue',
    'ValidationSeverity',
    'MissingDataStrategy',
    'QualityReport',
    'validate_ohlcv_data',
    'clean_ohlcv_data',
    'get_data_quality_score',

    # Volatility modeling
    'GARCHModel',
    'GARCHFitResult',
    'VolatilityForecast',
    'VolatilityRegime',
    'VolatilityRegimeDetector',
    'GARCHBacktestIntegration',
    'calculate_dynamic_var',
    'calculate_dynamic_stop_loss',
    'calculate_garch_adjusted_position_size',
    'ARCH_AVAILABLE',

    # Stress testing
    'StressTestFramework',
    'ScenarioGenerator',
    'StressTestResult',
    'StressTestReport',
    'HistoricalScenario',
    'HypotheticalScenario',
    'ReverseStressResult',
    'run_full_stress_test',
]
