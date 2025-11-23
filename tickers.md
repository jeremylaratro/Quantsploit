# Ticker Documentation

## Overview

This documentation provides information about supported tickers and symbols in the Quantsploit backtesting framework.

## Supported Symbols

The following symbols are commonly used for backtesting:

| Symbol | Name | Type | Market |
|--------|------|------|--------|
| SPY | SPDR S&P 500 ETF | ETF | US Equities |
| QQQ | Invesco QQQ Trust | ETF | US Tech |
| AAPL | Apple Inc. | Stock | NASDAQ |
| MSFT | Microsoft Corporation | Stock | NASDAQ |
| GOOGL | Alphabet Inc. | Stock | NASDAQ |
| AMZN | Amazon.com Inc. | Stock | NASDAQ |
| TSLA | Tesla Inc. | Stock | NASDAQ |
| META | Meta Platforms Inc. | Stock | NASDAQ |
| NVDA | NVIDIA Corporation | Stock | NASDAQ |

## Usage in Backtesting

To use these symbols in your backtests, specify them in the module configuration:

```python
# Single symbol
set SYMBOL AAPL

# Multiple symbols (comma-separated)
set SYMBOLS SPY, QQQ, AAPL, MSFT
```

## Symbol Categories

### Large Cap Tech
- **AAPL** - Consumer electronics and services
- **MSFT** - Software and cloud computing
- **GOOGL** - Internet services and advertising
- **META** - Social media and virtual reality

### Growth Stocks
- **TSLA** - Electric vehicles and energy
- **NVDA** - Graphics processing and AI chips
- **AMZN** - E-commerce and cloud services

### Market Indices
- **SPY** - Broad market exposure (S&P 500)
- **QQQ** - Tech-focused exposure (NASDAQ-100)

## Data Requirements

Each symbol requires:
- Historical price data (OHLCV)
- Minimum lookback period: 200 days for most strategies
- Data source: Yahoo Finance (via yfinance)

## Best Practices

1. **Diversification**: Use a mix of symbols from different sectors
2. **Liquidity**: Stick to high-volume symbols for accurate backtests
3. **Data Quality**: Verify data availability before running comprehensive backtests
4. **Timeframe**: Ensure sufficient historical data for your chosen backtest period

## Adding Custom Symbols

You can add any symbol supported by Yahoo Finance. Simply set it in your module:

```bash
set SYMBOLS YOUR_SYMBOL_1, YOUR_SYMBOL_2
```

> **Note**: Ensure the symbol has sufficient historical data for your backtest period.

## See Also

- [Comprehensive Backtest Guide](docs/COMPREHENSIVE_BACKTEST_GUIDE.md)
- [Advanced Strategies](ADVANCED_STRATEGIES.md)
- [README](README.md)
