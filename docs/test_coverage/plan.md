# Test Coverage Expansion Plan

## Baseline Assessment

**Date:** 2026-01-27
**Current Coverage:** 9% (1,196 / 13,926 statements covered)
**Target Coverage:** 95%
**Failing Tests:** 1 (test_default_config - expects 100000.0 but default is 1000.0)

## Current Test Infrastructure

- **Framework:** pytest with pytest-cov, pytest-mock, pytest-asyncio
- **Existing Tests:** 41 tests in 2 test files
- **Fixtures:** Comprehensive OHLCV data generators in `conftest.py`
- **Test Location:** `tests/` directory

## Coverage Analysis by Module

### Priority 1: Core Modules (Critical Infrastructure)

| Module | Statements | Covered | Coverage | Priority |
|--------|------------|---------|----------|----------|
| `core/database.py` | 86 | 16 | 19% | HIGH |
| `core/framework.py` | 100 | 25 | 25% | HIGH |
| `core/module.py` | 76 | 31 | 41% | HIGH |
| `core/session.py` | 35 | 14 | 40% | HIGH |
| **Subtotal** | **297** | **86** | **29%** | |

### Priority 2: Utility Modules (Foundation)

| Module | Statements | Covered | Coverage | Priority |
|--------|------------|---------|----------|----------|
| `utils/backtesting.py` | 314 | 273 | 87% | MEDIUM |
| `utils/data_fetcher.py` | 89 | 14 | 16% | HIGH |
| `utils/helpers.py` | 43 | 25 | 58% | MEDIUM |
| `utils/ta_compat.py` | 101 | 0 | 0% | HIGH |
| `utils/sample_data.py` | 58 | 15 | 26% | MEDIUM |
| `utils/statistical_analyzer.py` | 170 | 0 | 0% | HIGH |
| `utils/data_validation.py` | 477 | 77 | 16% | HIGH |
| `utils/logging_config.py` | 418 | 125 | 30% | MEDIUM |
| `utils/monte_carlo.py` | 451 | 92 | 20% | HIGH |
| `utils/options_greeks.py` | 651 | 0 | 0% | HIGH |
| `utils/portfolio_optimizer.py` | 484 | 72 | 15% | HIGH |
| `utils/position_sizing.py` | 299 | 48 | 16% | HIGH |
| `utils/stress_testing.py` | 593 | 105 | 18% | HIGH |
| `utils/ticker_validator.py` | 102 | 22 | 22% | HIGH |
| `utils/transaction_costs.py` | 296 | 85 | 29% | HIGH |
| `utils/volatility_models.py` | 478 | 81 | 17% | HIGH |
| `utils/walk_forward.py` | 275 | 55 | 20% | HIGH |
| `utils/comprehensive_backtest.py` | 625 | 0 | 0% | HIGH |
| **Subtotal** | **5,924** | **1,089** | **18%** | |

### Priority 3: Analysis Modules

| Module | Statements | Covered | Coverage | Priority |
|--------|------------|---------|----------|----------|
| `modules/analysis/advanced_filter.py` | 250 | 0 | 0% | HIGH |
| `modules/analysis/comprehensive_strategy_backtest.py` | 133 | 0 | 0% | HIGH |
| `modules/analysis/meta_analysis.py` | 239 | 0 | 0% | HIGH |
| `modules/analysis/pattern_recognition.py` | 177 | 0 | 0% | HIGH |
| `modules/analysis/period_analyzer.py` | 261 | 0 | 0% | HIGH |
| `modules/analysis/reddit_sentiment.py` | 522 | 0 | 0% | MEDIUM |
| `modules/analysis/sector_deep_dive.py` | 211 | 0 | 0% | HIGH |
| `modules/analysis/signal_aggregator.py` | 242 | 0 | 0% | HIGH |
| `modules/analysis/stock_analyzer.py` | 191 | 0 | 0% | HIGH |
| `modules/analysis/strategy_comparator.py` | 245 | 0 | 0% | HIGH |
| `modules/analysis/technical_indicators.py` | 95 | 0 | 0% | HIGH |
| **Subtotal** | **2,566** | **0** | **0%** | |

### Priority 4: Strategy Modules

| Module | Statements | Covered | Coverage | Priority |
|--------|------------|---------|----------|----------|
| `modules/strategies/hmm_regime_detection.py` | 224 | 0 | 0% | MEDIUM |
| `modules/strategies/kalman_adaptive.py` | 189 | 0 | 0% | MEDIUM |
| `modules/strategies/mean_reversion.py` | 123 | 0 | 0% | HIGH |
| `modules/strategies/ml_swing_trading.py` | 195 | 0 | 0% | MEDIUM |
| `modules/strategies/momentum_signals.py` | 140 | 0 | 0% | HIGH |
| `modules/strategies/multifactor_scoring.py` | 228 | 0 | 0% | MEDIUM |
| `modules/strategies/options_spreads.py` | 189 | 0 | 0% | MEDIUM |
| `modules/strategies/options_volatility.py` | 161 | 0 | 0% | MEDIUM |
| `modules/strategies/pairs_trading.py` | 225 | 0 | 0% | MEDIUM |
| `modules/strategies/reddit_sentiment_strategy.py` | 180 | 0 | 0% | LOW |
| `modules/strategies/sma_crossover.py` | 71 | 0 | 0% | HIGH |
| `modules/strategies/volume_profile_swing.py` | 191 | 0 | 0% | MEDIUM |
| **Subtotal** | **2,116** | **0** | **0%** | |

### Priority 5: Scanner Modules

| Module | Statements | Covered | Coverage | Priority |
|--------|------------|---------|----------|----------|
| `modules/scanners/bulk_screener.py` | 145 | 0 | 0% | HIGH |
| `modules/scanners/price_momentum.py` | 60 | 0 | 0% | HIGH |
| `modules/scanners/top_movers.py` | 154 | 0 | 0% | HIGH |
| **Subtotal** | **359** | **0** | **0%** | |

### Priority 6: Options Modules

| Module | Statements | Covered | Coverage | Priority |
|--------|------------|---------|----------|----------|
| `modules/options/options_analyzer.py` | 76 | 0 | 0% | MEDIUM |
| **Subtotal** | **76** | **0** | **0%** | |

### Priority 7: UI Modules

| Module | Statements | Covered | Coverage | Priority |
|--------|------------|---------|----------|----------|
| `ui/commands.py` | 433 | 0 | 0% | MEDIUM |
| `ui/console.py` | 43 | 0 | 0% | MEDIUM |
| `ui/display.py` | 106 | 0 | 0% | MEDIUM |
| **Subtotal** | **582** | **0** | **0%** | |

### Priority 8: Live Trading Modules (Lower Priority - Complex Mocking Required)

| Module | Statements | Covered | Coverage | Priority |
|--------|------------|---------|----------|----------|
| `live/broker_interface.py` | 905 | 0 | 0% | LOW |
| `live/realtime_data.py` | 908 | 0 | 0% | LOW |
| **Subtotal** | **1,813** | **0** | **0%** | |

### Priority 9: Webserver Module

| Module | Statements | Covered | Coverage | Priority |
|--------|------------|---------|----------|----------|
| `modules/webserver/webserver_manager.py` | 145 | 0 | 0% | LOW |
| **Subtotal** | **145** | **0** | **0%** | |

## Implementation Order

### Phase 1: Foundation (Core + Key Utilities)
- [x] Fix failing test_default_config test
- [ ] `tests/test_core_module.py` - Base module class
- [ ] `tests/test_core_framework.py` - Framework orchestration
- [ ] `tests/test_core_session.py` - Session management
- [ ] `tests/test_core_database.py` - Database operations
- [ ] `tests/test_technical_indicators.py` - TA calculations
- [ ] `tests/test_data_fetcher.py` - Data retrieval
- [ ] `tests/test_data_validation.py` - Input validation
- [ ] `tests/test_ta_compat.py` - TA compatibility layer

### Phase 2: Statistical & Risk Tools
- [ ] `tests/test_statistical_analyzer.py` - Statistical functions
- [ ] `tests/test_monte_carlo.py` - Monte Carlo simulations
- [ ] `tests/test_volatility_models.py` - Volatility calculations
- [ ] `tests/test_stress_testing.py` - Stress tests
- [ ] `tests/test_portfolio_optimizer.py` - Portfolio optimization
- [ ] `tests/test_position_sizing.py` - Position sizing
- [ ] `tests/test_transaction_costs.py` - Transaction cost models
- [ ] `tests/test_walk_forward.py` - Walk-forward analysis

### Phase 3: Options & Advanced Analytics
- [ ] `tests/test_options_greeks.py` - Greeks calculations
- [ ] `tests/test_options_analyzer.py` - Options analysis
- [ ] `tests/test_comprehensive_backtest.py` - Full backtest engine

### Phase 4: Analysis Modules
- [ ] `tests/test_analysis_stock_analyzer.py` - Stock analysis
- [ ] `tests/test_analysis_pattern_recognition.py` - Pattern detection
- [ ] `tests/test_analysis_signal_aggregator.py` - Signal aggregation
- [ ] `tests/test_analysis_advanced_filter.py` - Advanced filtering
- [ ] `tests/test_analysis_period_analyzer.py` - Period analysis
- [ ] `tests/test_analysis_sector_deep_dive.py` - Sector analysis
- [ ] `tests/test_analysis_meta_analysis.py` - Meta analysis
- [ ] `tests/test_analysis_strategy_comparator.py` - Strategy comparison

### Phase 5: Strategy Modules
- [ ] `tests/test_strategy_sma_crossover.py` - SMA crossover
- [ ] `tests/test_strategy_mean_reversion.py` - Mean reversion
- [ ] `tests/test_strategy_momentum_signals.py` - Momentum signals
- [ ] `tests/test_strategy_multifactor.py` - Multifactor scoring
- [ ] `tests/test_strategy_hmm_regime.py` - HMM regime detection
- [ ] `tests/test_strategy_kalman.py` - Kalman filter
- [ ] `tests/test_strategy_ml_swing.py` - ML swing trading
- [ ] `tests/test_strategy_pairs_trading.py` - Pairs trading
- [ ] `tests/test_strategy_options.py` - Options strategies
- [ ] `tests/test_strategy_volume_profile.py` - Volume profile

### Phase 6: Scanners
- [ ] `tests/test_scanner_bulk_screener.py` - Bulk screening
- [ ] `tests/test_scanner_price_momentum.py` - Price momentum
- [ ] `tests/test_scanner_top_movers.py` - Top movers

### Phase 7: UI & Display
- [ ] `tests/test_ui_commands.py` - Command handling
- [ ] `tests/test_ui_console.py` - Console I/O
- [ ] `tests/test_ui_display.py` - Display formatting

### Phase 8: Live Trading (Mocked)
- [ ] `tests/test_live_broker_interface.py` - Broker API mocking
- [ ] `tests/test_live_realtime_data.py` - Realtime data mocking

### Phase 9: Webserver
- [ ] `tests/test_webserver_manager.py` - Webserver management

## Testing Guidelines

### Test Structure
- Each test file should have classes grouping related tests
- Use descriptive test names: `test_<function>_<scenario>`
- Include docstrings explaining test purpose

### Required Test Types Per Module
1. **Happy path tests** - Normal operation
2. **Edge case tests** - Boundary conditions, empty inputs, etc.
3. **Error handling tests** - Invalid inputs, exceptions
4. **Integration tests** - Module interactions where appropriate

### Mocking Strategy
- Mock external API calls (yfinance, Reddit, etc.)
- Mock file I/O operations
- Mock database connections
- Use fixtures for common test data

### Coverage Targets by Module Type
- **Core modules:** 95%+
- **Utility modules:** 90%+
- **Strategy modules:** 85%+
- **UI modules:** 80%+
- **Live trading:** 70%+ (complex mocking)

## Estimated Test Counts

| Category | Modules | Est. Tests |
|----------|---------|------------|
| Core | 4 | 60 |
| Utilities | 18 | 300 |
| Analysis | 11 | 180 |
| Strategies | 12 | 200 |
| Scanners | 3 | 50 |
| Options | 1 | 30 |
| UI | 3 | 60 |
| Live | 2 | 80 |
| Webserver | 1 | 20 |
| **Total** | **55** | **~980** |

## Success Criteria

- All tests pass (0 failures)
- Overall coverage >= 95%
- No critical modules below 90% coverage
- All edge cases documented and tested
- CI pipeline integration ready
