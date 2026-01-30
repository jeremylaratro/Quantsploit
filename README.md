# Quantsploit

```
╔════════════════════════════════════════════════════════════════════════╗
║                                                                        ║
║    ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗██████╗ ██╗     ║
║   ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██║     ║
║   ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ███████╗██████╔╝██║     ║
║   ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ╚════██║██╔═══╝ ██║     ║
║   ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ███████║██║     ███████╗║
║    ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝     ╚══════╝║
║                                                                        ║
║                        EXPLOIT THE MARKET                              ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
```

**Quantitative Analysis Trading Framework with Interactive TUI**

Quantsploit is a modular quantitative trading framework inspired by penetration testing tools like Metasploit. It provides an interactive command-line interface for running financial analysis modules, algorithms, and trading strategies with a comprehensive web-based analytics dashboard.

## Features

- **Interactive TUI** - Metasploit-style command interface with auto-completion and session management
- **Modular Architecture** - Plugin system with 30+ modules for easy extension
- **Technical Analysis** - RSI, MACD, SMA, EMA, Bollinger Bands, Kalman filters, and more
- **Market Scanners** - Scan multiple stocks for momentum, volume, and patterns in parallel
- **Options Analysis** - Options chains, first and second-order Greeks, IV surface modeling
- **Strategy Backtesting** - 19 trading strategies with comprehensive performance metrics
- **Risk Analytics** - Risk parity, GARCH volatility, VaR/CVaR, and portfolio optimization
- **Analytics Dashboard** - Web-based visualization with interactive charts and multi-run comparison
- **Data Caching** - SQLite database for efficient data management
- **Watchlist Management** - Track and analyze favorite symbols

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jeremylaratro/Quantsploit.git
cd Quantsploit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Usage

### Starting Quantsploit

Run the framework:
```bash
python -m quantsploit.main
```

Or if installed:
```bash
quantsploit
```

### Basic Commands

| Command | Description | Example |
|---------|-------------|---------|
| `help` | Show available commands | `help` |
| `show modules` | List all available modules | `show modules` |
| `search <query>` | Search for modules | `search RSI` |
| `use <module>` | Load a module | `use analysis/technical_indicators` |
| `show options` | Display module options | `show options` |
| `set <OPTION> <value>` | Set module option | `set SYMBOL AAPL` |
| `run` | Execute current module | `run` |
| `back` | Unload current module | `back` |
| `quote <SYMBOL>` | Get real-time quote | `quote TSLA` |
| `watchlist add <SYMBOL>` | Add to watchlist | `watchlist add AAPL` |
| `webserver start` | Launch analytics dashboard | `webserver start --port 5000` |
| `analyze <type>` | Analyze backtest results | `analyze stock AAPL` |
| `compare` | Compare strategies | `compare sma_crossover mean_reversion` |
| `exit` | Exit Quantsploit | `exit` |

## Example Workflows

### 1. Technical Analysis on a Stock

```
quantsploit > use analysis/technical_indicators
quantsploit (Technical Indicators) > set SYMBOL AAPL
quantsploit (Technical Indicators) > set PERIOD 6mo
quantsploit (Technical Indicators) > run
```

This calculates RSI, MACD, SMA, EMA, and Bollinger Bands for Apple stock over the last 6 months.

### 2. Scan Multiple Stocks for Momentum

```
quantsploit > use scanners/price_momentum
quantsploit (Price Momentum Scanner) > set SYMBOLS AAPL,MSFT,GOOGL,TSLA,NVDA
quantsploit (Price Momentum Scanner) > set MIN_GAIN_PCT 3.0
quantsploit (Price Momentum Scanner) > run
```

### 3. Analyze Options Chain

```
quantsploit > use options/options_analyzer
quantsploit (Options Analyzer) > set SYMBOL SPY
quantsploit (Options Analyzer) > set MIN_VOLUME 100
quantsploit (Options Analyzer) > run
```

### 4. Backtest a Trading Strategy

```
quantsploit > use strategies/sma_crossover
quantsploit (SMA Crossover Strategy) > set SYMBOL AAPL
quantsploit (SMA Crossover Strategy) > set PERIOD 1y
quantsploit (SMA Crossover Strategy) > set FAST_PERIOD 10
quantsploit (SMA Crossover Strategy) > set SLOW_PERIOD 30
quantsploit (SMA Crossover Strategy) > set INITIAL_CAPITAL 10000
quantsploit (SMA Crossover Strategy) > run
```

## Available Modules

### Analysis Modules

| Module | Description |
|--------|-------------|
| `technical_indicators` | RSI, MACD, SMA, EMA, Bollinger Bands, and more |
| `pattern_recognition` | Candlestick and chart pattern detection with signals |
| `signal_aggregator` | Multi-strategy consensus buy/sell signals |
| `stock_analyzer` | Fundamental and technical stock analysis |
| `period_analyzer` | Performance analysis across time periods |
| `reddit_sentiment` | Reddit scraping and sentiment analysis |
| `meta_analysis` | Cross-strategy comparative analysis |
| `strategy_comparator` | Head-to-head strategy performance comparison |

### Scanner Modules

| Module | Description |
|--------|-------------|
| `price_momentum` | Multi-stock momentum and volume pattern scanning |
| `bulk_screener` | High-performance parallel screening (SP500, NASDAQ100) |
| `top_movers` | Top gainers, momentum leaders, multi-criterion ranking |

### Options Modules

| Module | Description |
|--------|-------------|
| `options_analyzer` | Options chain analysis, Greeks calculation, opportunity identification |

### Strategy Modules (19 Total)

#### Core Strategies
| Strategy | Description |
|----------|-------------|
| `sma_crossover` | Simple Moving Average crossover signals |
| `mean_reversion` | Z-score and Bollinger Bands mean reversion |
| `momentum_signals` | Multi-period momentum and trend following |
| `multifactor_scoring` | Comprehensive multi-factor quantitative model |

#### Advanced Strategies
| Strategy | Description |
|----------|-------------|
| `kalman_adaptive` | Kalman filter adaptive trend following |
| `volume_profile_swing` | Volume profile-based swing trading |
| `hmm_regime_detection` | Hidden Markov Model market regime detection |
| `ml_swing_trading` | Machine learning swing trading signals |
| `pairs_trading` | Statistical arbitrage pairs trading |
| `options_volatility` | Options volatility-based trading |
| `options_spreads` | Options spread strategies |
| `reddit_sentiment_strategy` | Reddit sentiment-driven signals |

#### v0.2.0 Strategies
| Strategy | Description |
|----------|-------------|
| `risk_parity` | Dynamic risk-parity allocation with GARCH and HRP |
| `volatility_breakout` | Bollinger Band squeeze breakout detection |
| `fama_french` | Three-factor model implementation |
| `earnings_momentum` | Earnings surprise momentum (requires earnings data) |
| `adaptive_allocation` | Dynamic allocation with market regime detection |
| `options_vol_arb` | Implied vs realized volatility arbitrage |
| `vwap_execution` | VWAP execution optimization (requires intraday data) |

## Advanced Features

### Risk Parity and Portfolio Optimization

Quantsploit includes institutional-grade portfolio optimization:

- **Risk Parity Targeted** - Volatility targeting with optional leverage
- **Leveraged Risk Parity** - Bridgewater-style leverage on low-volatility assets
- **Risk Parity GARCH** - GARCH volatility forecasting integration
- **Hierarchical Risk Parity** - HRP with min/max weight constraints
- **Dynamic Risk Budget** - Regime-dependent risk allocation

### Options Analytics

Extended options analysis capabilities:

- **First-Order Greeks** - Delta, Gamma, Theta, Vega, Rho
- **Second-Order Greeks** - Vanna, Volga, Charm, Veta, Speed, Zomma, Color, Ultima
- **IV Surface Builder** - SVI parameterization for volatility surface construction
- **Binomial Tree Pricing** - Cox-Ross-Rubinstein model for American options
- **Options Risk Dashboard** - Portfolio-level risk analysis, stress testing, VaR estimation

### Bulk Analysis

- Analyze 100+ stocks in parallel with the Advanced Bulk Screener
- Pattern recognition with 10+ candlestick and chart patterns
- Multi-factor scoring combining momentum, technical, volatility, and volume factors
- Signal aggregation from 5+ strategies with confidence scoring

## Analytics Dashboard

Quantsploit includes a comprehensive web-based dashboard for visualizing and analyzing backtest results.

### Dashboard Features

- **Interactive Charts** - Performance metrics, returns, and risk visualization
- **Period Analysis** - Compare performance across quarters and custom time periods
- **Strategy Rankings** - Rank by Sharpe ratio, return, win rate, signal accuracy
- **Risk vs Return** - Interactive scatter plots and heatmaps
- **Multi-Run Comparison** - Compare up to 5 different backtest runs
- **Correlation Analysis** - Inter-strategy correlation matrices
- **Rolling Metrics** - Window-based performance analysis
- **Export Capabilities** - Save charts and generate reports

### Quick Start

1. Run a comprehensive backtest:
```bash
python run_comprehensive_backtest.py --symbols AAPL,MSFT,GOOGL
```

2. Launch the dashboard:
```bash
# From within Quantsploit TUI
quantsploit > webserver start --port 5000

# Or directly
python dashboard/app.py
```

3. Open your browser to `http://localhost:5000`

### Dashboard API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/runs` | List available backtest runs |
| `/api/summary/<timestamp>` | Summary statistics for a run |
| `/api/detailed/<timestamp>` | Detailed backtest results |
| `/api/quarterly/<timestamp>` | Period-based comparisons |
| `/api/strategies/<timestamp>` | Strategy-level analysis |
| `/api/symbols/<timestamp>` | Symbol performance data |
| `/api/heatmap/<timestamp>` | Strategy x period heatmap |

## Creating Custom Modules

All modules inherit from `BaseModule` and implement the following structure:

```python
from quantsploit.core.module import BaseModule
from typing import Dict, Any

class MyCustomModule(BaseModule):
    @property
    def name(self) -> str:
        return "My Custom Module"

    @property
    def description(self) -> str:
        return "Description of what this module does"

    @property
    def author(self) -> str:
        return "Your Name"

    @property
    def category(self) -> str:
        return "analysis"  # or scanner, options, strategy

    def _init_options(self):
        super()._init_options()
        self.options.update({
            "MY_OPTION": {
                "value": None,
                "required": True,
                "description": "My custom option"
            }
        })

    def run(self) -> Dict[str, Any]:
        # Your module logic here
        symbol = self.get_option("SYMBOL")

        # Return results
        return {
            "symbol": symbol,
            "result": "some value"
        }
```

Save your module in the appropriate directory:
- `quantsploit/modules/analysis/` - For analysis modules
- `quantsploit/modules/scanners/` - For scanner modules
- `quantsploit/modules/options/` - For options modules
- `quantsploit/modules/strategies/` - For strategy modules

## Configuration

Edit `config.yaml` to customize:

- Database location
- Data caching settings
- Display preferences
- Module paths

## Architecture

```
quantsploit/
├── core/
│   ├── framework.py       # Main framework engine
│   ├── module.py          # Base module class
│   ├── session.py         # Session management
│   └── database.py        # SQLite database
├── modules/
│   ├── analysis/          # Technical analysis modules (11)
│   ├── scanners/          # Market scanner modules (3)
│   ├── options/           # Options analysis modules (1)
│   └── strategies/        # Trading strategy modules (19)
├── ui/
│   ├── console.py         # Interactive TUI
│   ├── commands.py        # Command handlers
│   └── display.py         # Display utilities
├── utils/
│   ├── data_fetcher.py    # Market data fetching
│   ├── backtesting.py     # Backtesting engine
│   ├── options_greeks.py  # Options analytics
│   ├── risk_parity.py     # Portfolio optimization
│   └── volatility_models.py # Volatility modeling
└── dashboard/
    ├── app.py             # Flask web application
    ├── strategy_api.py    # Strategy execution API
    └── templates/         # Dashboard templates
```

## Data Sources

Quantsploit uses:
- **yfinance** - Yahoo Finance API for market data
- **pandas-ta** - Technical analysis indicators
- **py_vollib** - Options pricing and Greeks
- **scipy** - Statistical computations and optimization
- **scikit-learn** - Machine learning models

## Database

The framework uses SQLite to cache:
- Market data (price/volume)
- Analysis results
- Watchlist symbols
- Backtest results

Database location: `./quantsploit.db` (configurable)

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch
3. Add your module or enhancement
4. Submit a pull request

## Disclaimer

**This framework is for educational and research purposes only. It is not financial advice. Trading stocks and options carries risk. Always do your own research and consult with a financial advisor before making investment decisions.**

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Inspired by Metasploit's modular framework architecture
- Built with Python, pandas, yfinance, and other open-source libraries
