# Comprehensive Strategy Backtesting Guide

This guide explains how to use Quantsploit's comprehensive backtesting system to evaluate multiple trading strategies across different time periods and identify the most accurate signal generators.

## Overview

The comprehensive backtesting system allows you to:
- Test all available strategies simultaneously
- Analyze performance across multiple time periods (1-3 years, 6-month windows)
- Compare strategies head-to-head
- Identify which strategies produce the most accurate signals
- Generate detailed performance reports

## Quick Start

### Method 1: Standalone Script (Recommended)

```bash
# Basic usage - test 5 major stocks
python run_comprehensive_backtest.py

# Custom symbols
python run_comprehensive_backtest.py --symbols AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA

# Custom capital and commission
python run_comprehensive_backtest.py \
    --symbols SPY,QQQ,IWM \
    --capital 50000 \
    --commission 0.001

# Custom output directory
python run_comprehensive_backtest.py \
    --symbols AAPL,MSFT \
    --output ./my_results

# Quick test mode (faster, single symbol)
python run_comprehensive_backtest.py --quick
```

### Method 2: Using the Quantsploit Module System

```bash
# Launch Quantsploit
python main.py

# Use the comprehensive backtest module
use analysis/comprehensive_strategy_backtest
set SYMBOLS AAPL,MSFT,GOOGL,TSLA,SPY
set OUTPUT_DIR ./backtest_results
set INITIAL_CAPITAL 100000
set COMMISSION 0.001
run
```

### Method 3: Python API

```python
from quantsploit.utils.comprehensive_backtest import run_comprehensive_analysis

# Run analysis
results_df, summary = run_comprehensive_analysis(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY'],
    output_dir='./backtest_results'
)

# Access results
print(summary['overall_stats'])
print(summary['best_by_total_return'])
```

## Available Strategies

The system automatically tests these strategies:

1. **SMA Crossover (20/50)** - Classic trend-following with 20/50 day moving averages
2. **SMA Crossover (10/30)** - Faster trend-following with 10/30 day moving averages
3. **Mean Reversion (20 day)** - Standard mean reversion based on 20-day z-scores
4. **Mean Reversion Aggressive** - Faster mean reversion with 10-day lookback
5. **Momentum (10/20/50)** - Multi-period momentum strategy
6. **Momentum Aggressive** - Faster momentum with 5/10/20 day periods

## Test Periods

The system automatically generates comprehensive test periods:

### Full Period Tests
- **1 Year** - Last 365 days
- **2 Year** - Last 730 days
- **3 Year** - Last 1,095 days

### Rolling 6-Month Periods
- **Period 1** - Most recent 6 months
- **Period 2** - 6-12 months ago
- **Period 3** - 12-18 months ago
- **Period 4** - 18-24 months ago

This gives you insight into strategy performance across different market conditions and timeframes.

## Performance Metrics

Each backtest calculates comprehensive metrics:

### Return Metrics
- **Total Return** - Overall percentage return
- **Annual Return** - Annualized return
- **Buy & Hold Return** - Passive benchmark return
- **Excess Return** - Strategy return minus buy & hold

### Risk Metrics
- **Sharpe Ratio** - Risk-adjusted return (higher is better)
- **Sortino Ratio** - Downside risk-adjusted return
- **Max Drawdown** - Largest peak-to-trough decline
- **Volatility** - Annualized standard deviation of returns

### Trade Metrics
- **Total Trades** - Number of completed trades
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Ratio of gross profits to gross losses
- **Average Win/Loss** - Average profit and loss per trade

### Signal Accuracy
- **Correct Signals** - Number of profitable trades
- **Total Signals** - Total number of trades taken
- **Signal Accuracy** - Percentage of correct signals

## Output Files

The system generates three files in your output directory:

### 1. Detailed Results CSV
`detailed_results_YYYYMMDD_HHMMSS.csv`

Contains every backtest result with all metrics. Great for:
- Excel analysis
- Custom visualizations
- Statistical analysis
- Further data processing

### 2. Summary JSON
`summary_YYYYMMDD_HHMMSS.json`

Structured summary with:
- Best strategies by different metrics
- Performance by time period
- Performance by symbol
- Overall statistics

### 3. Markdown Report
`report_YYYYMMDD_HHMMSS.md`

Human-readable report with:
- Executive summary
- Top 10 strategies by total return
- Top 10 by Sharpe ratio
- Top 10 by signal accuracy
- Tables and rankings

## Interpreting Results

### Finding the Best Strategies

The system ranks strategies by multiple criteria:

1. **Best by Total Return** - Highest absolute profits
   - Good for: Identifying maximum profit potential
   - Caution: May involve higher risk

2. **Best by Sharpe Ratio** - Best risk-adjusted returns
   - Good for: Balancing risk and reward
   - Recommended metric for most traders

3. **Best by Signal Accuracy** - Highest win rate
   - Good for: Consistent performance
   - Note: High accuracy doesn't always mean high returns

4. **Best Excess Return** - Outperformance vs buy & hold
   - Good for: Validating strategy value
   - Critical metric for active trading

### Key Questions to Ask

1. **Consistency**: Does the strategy perform well across multiple time periods?
2. **Symbol Independence**: Does it work on different stocks or just one?
3. **Risk-Adjusted**: Is the return worth the risk (check Sharpe ratio)?
4. **Beating Benchmark**: Does it beat simple buy & hold?
5. **Trade Frequency**: Does it generate enough signals without overtrading?

### Red Flags

⚠️ **Warning Signs:**
- Very high returns in only one period (likely overfitting)
- Low Sharpe ratio despite high returns (excessive risk)
- Very few trades (insufficient data)
- Negative excess return (worse than buy & hold)
- High win rate but negative returns (small wins, large losses)

## Example Workflow

Here's a recommended workflow for identifying the best strategies:

### Step 1: Run Initial Test
```bash
# Test on diverse symbols
python run_comprehensive_backtest.py --symbols AAPL,MSFT,SPY,TSLA,JPM
```

### Step 2: Review Results
```bash
# Check the markdown report
cat backtest_results/report_*.md

# Or open in your favorite editor
```

### Step 3: Identify Top Performers
Look for strategies that:
- Appear in top 10 for multiple metrics
- Work across different symbols
- Consistent across time periods
- Sharpe ratio > 1.0
- Positive excess returns

### Step 4: Deep Dive
```bash
# Test winning strategies on more symbols
python run_comprehensive_backtest.py --symbols QQQ,IWM,DIA,XLF,XLE,XLK
```

### Step 5: Validate
- Check if top strategies remain consistent
- Verify performance across market sectors
- Test in different market conditions

## Advanced Usage

### Customizing Strategies

Edit `quantsploit/utils/comprehensive_backtest.py` to add custom strategies:

```python
# In ComprehensiveBacktester.__init__, add to self.strategies:
'my_custom_strategy': {
    'name': 'My Custom Strategy',
    'function': self.adapter.my_custom_strategy,
    'params': {'param1': value1, 'param2': value2}
}

# Then create the adapter function in StrategyAdapter class:
def my_custom_strategy(self, backtester, date, row, symbol, param1, param2):
    # Your strategy logic here
    pass
```

### Custom Time Periods

Modify `generate_test_periods()` method:

```python
# Add a specific date range
periods.append(TestPeriod(
    name='covid_crash',
    start_date='2020-02-01',
    end_date='2020-04-01',
    description='COVID-19 Market Crash'
))
```

### Analyzing Results Programmatically

```python
import pandas as pd

# Load results
df = pd.read_csv('backtest_results/detailed_results_*.csv')

# Find strategies that beat buy & hold
winners = df[df['excess_return'] > 0]

# Best strategy per symbol
best_per_symbol = df.loc[df.groupby('symbol')['total_return'].idxmax()]

# Strategies with Sharpe > 1.5
high_sharpe = df[df['sharpe_ratio'] > 1.5]

# Average performance by strategy
strategy_avg = df.groupby('strategy_name').agg({
    'total_return': 'mean',
    'sharpe_ratio': 'mean',
    'win_rate': 'mean',
    'signal_accuracy': 'mean'
}).sort_values('sharpe_ratio', ascending=False)
```

## Tips for Best Results

### Symbol Selection
- **Diversity**: Mix large-cap, mid-cap, small-cap
- **Sectors**: Include different market sectors
- **Liquidity**: Ensure symbols have good trading volume
- **ETFs**: Include SPY, QQQ for benchmark comparison

### Recommended Test Symbols
```python
# Diverse portfolio test
--symbols AAPL,MSFT,GOOGL,AMZN,TSLA,JPM,BA,WMT,SPY,QQQ

# Sector test
--symbols XLF,XLE,XLK,XLV,XLI,XLP,XLY,XLB,XLU

# Large-cap test
--symbols AAPL,MSFT,GOOGL,AMZN,META,TSLA,NVDA,BRK.B
```

### Performance Optimization
- Start with `--quick` mode for testing
- Use fewer symbols initially (3-5)
- Increase after validating setup
- Parallel processing coming in future updates

## Troubleshooting

### Common Issues

**Issue**: No results generated
- **Solution**: Check if symbols are valid, try with SPY first

**Issue**: Very poor performance across all strategies
- **Solution**: Check commission/slippage settings, ensure data is downloading correctly

**Issue**: Missing data errors
- **Solution**: Yahoo Finance API may be down, try again later or use different symbols

**Issue**: Script runs slowly
- **Solution**: Reduce number of symbols or use --quick mode

### Getting Help

- Check the logs for detailed error messages
- Review example outputs in `examples/` directory
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## Next Steps

After identifying your best strategies:

1. **Paper Trading**: Test in real-time with paper money
2. **Live Testing**: Start with small capital
3. **Monitoring**: Track actual vs backtested performance
4. **Optimization**: Fine-tune parameters based on results
5. **Diversification**: Combine multiple winning strategies

## Best Practices

✅ **Do:**
- Test on diverse symbols and time periods
- Focus on risk-adjusted returns (Sharpe ratio)
- Verify strategies beat buy & hold
- Consider transaction costs
- Look for consistency across periods

❌ **Don't:**
- Over-optimize on a single symbol
- Ignore risk metrics
- Use strategies with very few trades
- Trust results from a single time period
- Overlook market conditions during test periods

## Conclusion

The comprehensive backtesting system gives you powerful tools to:
- Objectively compare strategies
- Identify robust signal generators
- Make data-driven trading decisions
- Understand strategy strengths and weaknesses

Remember: **Past performance does not guarantee future results.** Use backtesting as one tool in your trading toolkit, combined with fundamental analysis, risk management, and sound judgment.

Happy backtesting! 📈
