# Quantsploit

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗██████╗║
║  ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝██╔══██║
║  ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ███████╗██████╔╝
║  ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ╚════██║██╔═══╝║
║  ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ███████║██║    ║
║   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝    ║
║                    EXPLOIT THE MARKET                         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

**Quantitative Analysis Trading Framework with Interactive TUI**

Quantsploit is a modular quantitative trading framework inspired by penetration testing tools like Metasploit. It provides an interactive command-line interface for running various financial analysis modules, algorithms, and trading strategies.

## Features

- 🖥️ **Interactive TUI** - Metasploit-style command interface with auto-completion
- 🔧 **Modular Architecture** - Plugin system for easy extension
- 📊 **Technical Analysis** - RSI, MACD, SMA, EMA, Bollinger Bands, and more
- 🔍 **Market Scanners** - Scan multiple stocks for momentum, volume, and patterns
- 📈 **Options Analysis** - Analyze options chains, Greeks, and opportunities
- 💼 **Strategy Backtesting** - Test and validate trading strategies
- 📉 **Analytics Dashboard** - Web-based visualization with interactive charts and comparisons
- 💾 **Data Caching** - SQLite database for efficient data management
- 📋 **Watchlist Management** - Track your favorite symbols
- 🎨 **Rich Output** - Beautiful tables and formatted results

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Quantsploit.git
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
| `exit` | Exit Quantsploit | `exit` |

## Example Workflows

### 1. Technical Analysis on a Stock

```
quantsploit > use analysis/technical_indicators
quantsploit (Technical Indicators) > set SYMBOL AAPL
quantsploit (Technical Indicators) > set PERIOD 6mo
quantsploit (Technical Indicators) > run
```

This will calculate RSI, MACD, SMA, EMA, and Bollinger Bands for Apple stock over the last 6 months.

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

- **technical_indicators** - Calculate technical indicators (RSI, MACD, SMA, EMA, Bollinger Bands)
- **pattern_recognition** - Detect candlestick and chart patterns with automated signals
- **signal_aggregator** - Aggregate multiple strategies for consensus buy/sell signals

### Scanner Modules

- **price_momentum** - Scan multiple stocks for price momentum and volume patterns
- **bulk_screener** - High-performance parallel screening of large stock universes (SP500, NASDAQ100)
- **top_movers** - Identify top gainers, momentum leaders, and rank stocks by multiple criteria

### Options Modules

- **options_analyzer** - Analyze options chain, calculate metrics, and identify opportunities

### Strategy Modules

- **sma_crossover** - Simple Moving Average crossover backtesting strategy
- **mean_reversion** - Statistical mean reversion with z-score and Bollinger Bands analysis
- **momentum_signals** - Advanced momentum and trend following with multiple confirmations
- **multifactor_scoring** - Comprehensive multi-factor quantitative scoring system

## 🔥 Advanced Features

Quantsploit includes cutting-edge quantitative algorithms for serious traders:

- **Bulk Analysis** - Analyze 100+ stocks in parallel with the Advanced Bulk Screener
- **Pattern Recognition** - Automated detection of 10+ candlestick and chart patterns
- **Mean Reversion** - Z-score analysis, percentile ranking, and reversion probability
- **Momentum Strategies** - Multi-period momentum, acceleration, and relative strength
- **Multi-Factor Models** - Combine momentum, technical, volatility, and volume factors
- **Signal Aggregation** - Consensus signals from 5+ strategies with confidence scoring
- **Top Movers** - Real-time rankings by gainers, momentum, breakouts, and quality

See [ADVANCED_STRATEGIES.md](ADVANCED_STRATEGIES.md) for detailed usage guide.

## 📊 Backtesting Analytics Dashboard

Quantsploit includes a comprehensive web-based dashboard for visualizing and analyzing backtest results:

### Features
- **Interactive Charts** - Visualize performance metrics, returns, and risk across strategies
- **Period Analysis** - Compare performance across quarters and custom time periods
- **Strategy Rankings** - Identify top-performing strategies with detailed metrics
- **Risk vs Return** - Interactive scatter plots and heatmaps
- **Multi-Run Comparison** - Compare up to 5 different backtest runs
- **Export Capabilities** - Save charts and generate reports

### Quick Start

1. Run a comprehensive backtest:
```bash
python run_comprehensive_backtest.py --symbols AAPL,MSFT,GOOGL
```

2. Launch the dashboard:
```bash
./start_dashboard.sh  # Linux/Mac
# or
start_dashboard.bat   # Windows
```

3. Open your browser to `http://localhost:5000`

See [DASHBOARD.md](DASHBOARD.md) for complete documentation and usage guide.

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
│   ├── framework.py      # Main framework engine
│   ├── module.py          # Base module class
│   ├── session.py         # Session management
│   └── database.py        # SQLite database
├── modules/
│   ├── analysis/          # Technical analysis modules
│   ├── scanners/          # Market scanner modules
│   ├── options/           # Options analysis modules
│   └── strategies/        # Trading strategy modules
├── ui/
│   ├── console.py         # Interactive TUI
│   ├── commands.py        # Command handlers
│   └── display.py         # Display utilities
└── utils/
    ├── data_fetcher.py    # Market data fetching
    └── helpers.py         # Utility functions
```

## Data Sources

Quantsploit uses:
- **yfinance** - Yahoo Finance API for market data
- **pandas-ta** - Technical analysis indicators
- **py_vollib** - Options pricing and Greeks

## Database

The framework uses SQLite to cache:
- Market data (price/volume)
- Analysis results
- Watchlist symbols

Database location: `./quantsploit.db` (configurable)

## Roadmap

- [ ] More technical indicators (Ichimoku, Fibonacci, etc.)
- [ ] Advanced options strategies (Iron Condor, Butterfly, etc.)
- [ ] Machine learning modules
- [ ] Real-time streaming data
- [ ] Visualization/charting
- [ ] Portfolio management
- [ ] Risk analysis modules
- [ ] Correlation analysis
- [ ] News sentiment analysis
- [ ] Backtesting framework enhancements

## Contributing

Contributions are welcome! Please:

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
- Built with Python, pandas, yfinance, and other excellent open-source libraries

---

**Happy Trading! 📈**