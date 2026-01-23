# Changelog

All notable changes to Quantsploit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-23

### Added - Risk Parity Enhancements

New advanced risk parity methods in `quantsploit/utils/portfolio_optimizer.py`:

- **`risk_parity_targeted()`** - Volatility targeting with optional leverage
  - Scale portfolio to target volatility level
  - Supports leveraged and deleveraged portfolios
  - Maintains risk parity allocation principles

- **`leveraged_risk_parity()`** - Bridgewater-style leverage on low-volatility assets
  - Automatically applies leverage to achieve higher returns
  - Maintains diversification benefits of risk parity
  - Includes leverage constraints and monitoring

- **`risk_parity_rebalance()`** - Transaction cost-aware rebalancing
  - Drift threshold-based rebalancing triggers
  - Minimizes unnecessary turnover
  - Accounts for transaction costs in optimization

- **`risk_parity_garch()`** - GARCH volatility forecasting integration
  - Forward-looking volatility estimates via GARCH(1,1)
  - EWMA fallback for robustness
  - More responsive to volatility regime changes

- **`hierarchical_risk_parity_constrained()`** - HRP with weight constraints
  - Hierarchical Risk Parity with min/max weight limits
  - Maintains HRP diversification benefits
  - Enforces practical portfolio constraints

- **`dynamic_risk_budget()`** - Regime-dependent risk allocation
  - Volatility regime detection (low/normal/high)
  - Automatic risk budget adjustment by regime
  - Defensive allocation in high-vol environments

### Added - Options Analytics Enhancements

Extended options Greeks and analytics in `quantsploit/utils/options_greeks.py`:

#### Second-Order Greeks (8 new metrics)
- **Vanna** - Sensitivity of delta to volatility changes (∂Δ/∂σ)
- **Volga (Vomma)** - Sensitivity of vega to volatility changes (∂²V/∂σ²)
- **Charm** - Delta decay over time (∂Δ/∂t)
- **Veta** - Vega decay over time (∂Vega/∂t)
- **Speed** - Third derivative of price with respect to spot (∂Γ/∂S)
- **Zomma** - Sensitivity of gamma to volatility (∂Γ/∂σ)
- **Color** - Sensitivity of gamma to time (∂Γ/∂t)
- **Ultima** - Third-order volatility sensitivity (∂³V/∂σ³)

#### New Features
- **`calculate_all_greeks_extended()`** - Returns comprehensive `GreeksResult` dataclass with all first and second-order Greeks
- **`BinomialTree` class** - Cox-Ross-Rubinstein (CRR) model for American options pricing
  - Early exercise detection
  - Configurable number of steps
  - Support for calls and puts
- **`IVSurfaceBuilder` class** - Implied Volatility surface construction
  - SVI (Stochastic Volatility Inspired) parameterization
  - Calibration from market option prices
  - Interpolation and visualization support
- **`OptionsRiskDashboard` class** - Portfolio-level options risk analytics
  - Aggregate Greeks across multiple positions
  - Stress testing (price, volatility, time scenarios)
  - Value-at-Risk (VaR) estimation
  - Risk exposure summaries

### Changed

- Enhanced `quantsploit/utils/__init__.py` with new module exports
- Updated `requirements.txt` with additional dependencies for advanced analytics

### Removed

- Deleted outdated documentation files (ADVANCED_STRATEGIES.md, ANALYSIS_GUIDE.md, DASHBOARD.md, README_QUICK_START.md)
- Removed legacy test files and debug scripts
- Cleaned up obsolete shell scripts and batch files
- Removed bundled multitasking package directory

## [0.1.0] - 2025-11-23

### Initial Release

- Basic quantitative trading framework
- Multiple strategy implementations
- Backtesting engine
- Interactive TUI console
- Web dashboard for results visualization
