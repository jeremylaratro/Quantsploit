# Quantsploit Analysis Tools - Complete Guide

## Overview

This guide covers the new comprehensive analysis framework designed to help you:
1. **Identify sectors with potential**
2. **Find individual stocks worth investigating**
3. **Compare strategies and discover which are most effective**
4. **Analyze specific stocks in detail**
5. **Make data-driven decisions** about stock and strategy selection

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Fixed: Large Standard Deviation Issues](#fixed-large-standard-deviation-issues)
3. [Stock Deep Dive Analysis](#stock-deep-dive-analysis)
4. [Strategy Comparison](#strategy-comparison)
5. [Enhanced Sector Analysis](#enhanced-sector-analysis)
6. [Time Period Analysis](#time-period-analysis)
7. [Advanced Filtering](#advanced-filtering)
8. [Complete Workflow Example](#complete-workflow-example)
9. [CLI Commands Reference](#cli-commands-reference)
10. [Python API Reference](#python-api-reference)

---

## Quick Start

### From the CLI (Interactive Console)

```bash
# Start Quantsploit
python main.py

# Run a comprehensive backtest first
quantsploit > use backtesting/comprehensive
quantsploit(comprehensive) > set SYMBOLS AAPL,NVDA,MSFT,GOOGL,TSLA
quantsploit(comprehensive) > run

# Now use the analysis tools

# 1. Analyze a specific stock
quantsploit > analyze stock AAPL

# 2. Compare two strategies
quantsploit > compare "SMA Crossover (20/50)" "Kalman Adaptive Filter" --stock AAPL

# 3. Filter results
quantsploit > filter --sector AI/Tech --min-sharpe 1.0 --top-n 10

# 4. Analyze a sector
quantsploit > analyze sector AI/Tech
```

### From Python Scripts

```python
from modules.analysis.stock_analyzer import StockAnalyzer
from modules.analysis.strategy_comparator import StrategyComparator
from modules.analysis.sector_deep_dive import SectorAnalyzer
from modules.analysis.advanced_filter import AdvancedFilter

# Load latest backtest results
analyzer = StockAnalyzer.from_timestamp('20251122_203908')

# Analyze AAPL across all strategies and periods
summary = analyzer.get_stock_summary('AAPL')
print(summary)
```

---

## Fixed: Large Standard Deviation Issues

### The Problem

Previous statistics calculations aggregated **all strategies together** without stratification:

```
Example from old system:
Period: 1yr
  Mean Return:    12.5%
  Std Dev:        45.2%  ← HUGE! Why?

Breakdown:
  SMA Crossover:         +8.0%  (conservative)
  Mean Reversion:       -16.9%  (conservative)
  Kalman (Sensitive):   +48.5%  (aggressive)

Standard deviation = huge because mixing apples and oranges!
```

### The Solution

New **Stratified Statistics** that separate strategies by risk class:

```
Period: 1yr (STRATIFIED)

CONSERVATIVE Strategies:
  Median: 5.2%
  Std:    8.5%  ← Much more reasonable!

MODERATE Strategies:
  Median: 12.1%
  Std:    14.3%

AGGRESSIVE Strategies:
  Median: 18.7%
  Std:    22.8%
```

**Key Features:**
- ✅ Stratification by risk class (conservative/moderate/aggressive)
- ✅ Robust statistics (median, MAD, IQR) instead of naive mean/stdev
- ✅ Outlier detection and removal
- ✅ Minimum sample size requirements
- ✅ Confidence intervals via bootstrapping
- ✅ Quality scores and reliability ratings

---

## Stock Deep Dive Analysis

### Purpose

Analyze a **single stock** across all strategies and time periods to answer:
- Which strategy works best for this stock?
- Is performance consistent across different time periods?
- What are the risk metrics?
- What's the optimal configuration for trading this stock?

### CLI Usage

```bash
quantsploit > analyze stock AAPL
quantsploit > analyze stock NVDA --timestamp 20251122_203908
```

### Python Usage

```python
from modules.analysis.stock_analyzer import StockAnalyzer

# Load analyzer
analyzer = StockAnalyzer.from_csv('backtest_results/detailed_results_20251122_203908.csv')

# Deep dive on AAPL
analysis = analyzer.analyze_stock('AAPL', min_trades=5)

print(f"Best Strategy: {analysis.best_strategy}")
print(f"Best Return: {analysis.best_combination_return:.2f}%")
print(f"Most Consistent: {analysis.most_consistent_strategy}")

# Get formatted report
summary = analyzer.get_stock_summary('AAPL')
print(summary)

# Compare multiple stocks
comparison = analyzer.compare_stocks(['AAPL', 'NVDA', 'MSFT', 'GOOGL'])
print(comparison)
```

### Example Output

```
======================================================================
STOCK DEEP DIVE: AAPL
======================================================================

OVERVIEW:
  Reliability:     High (Quality Score: 87.5/100)
  Total Tests:     90 (10 strategies × 9 periods)

PERFORMANCE:
  Average Return:    12.45%
  Median Return:     11.80%
  Best Return:       28.73%
  Worst Return:      -5.21%

RISK METRICS:
  Avg Sharpe Ratio:   1.23
  Avg Volatility:     18.50%
  Avg Max Drawdown:  -12.30%

BEST COMBINATION:
  Strategy:        Kalman Adaptive Filter
  Period:          1yr
  Return:          28.73%
  Sharpe Ratio:    1.85

MOST CONSISTENT:
  Strategy:        Multi-Factor Scoring
  CV:              0.421 (lower = more consistent)

──────────────────────────────────────────────────────────────────────
TOP 5 STRATEGIES (by Average Return):
──────────────────────────────────────────────────────────────────────

1. Kalman Adaptive Filter (aggressive)
   Avg Return:       18.45% (Median: 17.20%)
   Sharpe Ratio:      1.45
   Win Rate:         72.5%
   Success Rate:     88.9% (8/9 periods)
   Consistency:       0.782
   Best Period:      1yr (+28.73%)
   95% CI:           [14.21%, 22.69%]
   Reliability:      High

2. Multi-Factor Scoring (moderate)
   Avg Return:       15.23% (Median: 15.01%)
   Sharpe Ratio:      1.38
   Win Rate:         68.2%
   Success Rate:     100.0% (9/9 periods)
   Consistency:       0.891  ← Very consistent!
   Best Period:      2yr (+18.45%)
   95% CI:           [13.12%, 17.34%]
   Reliability:      High

...
```

---

## Strategy Comparison

### Purpose

Compare 2+ strategies **head-to-head** with statistical significance testing.

### CLI Usage

```bash
# Compare two strategies globally
quantsploit > compare "SMA Crossover (20/50)" "Kalman Adaptive Filter"

# Compare for a specific stock
quantsploit > compare "SMA Crossover (20/50)" "Momentum (10/20/50)" --stock AAPL

# Compare for a sector
quantsploit > compare "Multi-Factor Scoring" "HMM Regime Detection" --sector AI/Tech
```

### Python Usage

```python
from modules.analysis.strategy_comparator import StrategyComparator

# Load comparator
comparator = StrategyComparator.from_csv('backtest_results/detailed_results_20251122_203908.csv')

# Compare two strategies
result = comparator.compare_two_strategies(
    'SMA Crossover (20/50)',
    'Kalman Adaptive Filter',
    stock='AAPL'  # Optional: filter by stock
)

print(f"Winner: {result.winner}")
print(f"Return Difference: {result.return_diff:+.2f}%")
print(f"P-Value: {result.p_value:.4f}")
print(f"Significant: {result.is_significant}")

# Compare 3+ strategies
multi_result = comparator.compare_multiple_strategies(
    ['SMA Crossover (20/50)', 'Kalman Adaptive Filter', 'Multi-Factor Scoring'],
    stock='NVDA'
)

print(f"Overall Winner: {multi_result.overall_winner}")
```

### Example Output

```
======================================================================
STRATEGY COMPARISON
======================================================================

SMA Crossover (20/50)  vs  Kalman Adaptive Filter

PERFORMANCE:
  Return Difference:    +10.25% (Kalman - SMA)
  Sharpe Difference:    +0.52
  Win Rate Difference:  +8.3%

WINNER:
  By Return:            Kalman Adaptive Filter (High confidence)
  By Risk-Adj Return:   Kalman Adaptive Filter

STATISTICAL TEST (Mann-Whitney U):
  Test Statistic:       1234.56
  P-Value:              0.0023
  Significant:          Yes (p < 0.05)
  Interpretation:       Highly significant difference (p < 0.01)

HEAD-TO-HEAD (on same stocks & periods):
  Kalman wins:          42
  SMA wins:             18
  Ties:                 2

======================================================================
```

---

## Enhanced Sector Analysis

### Purpose

Analyze sectors with **stratified statistics** to fix the large standard deviation problem.

### CLI Usage

```bash
# Analyze a specific sector
quantsploit > analyze sector AI/Tech

# Compare all sectors
# (Use Python API for now)
```

### Python Usage

```python
from modules.analysis.sector_deep_dive import SectorAnalyzer

# Load analyzer
analyzer = SectorAnalyzer.from_csv('backtest_results/detailed_results_20251122_203908.csv')

# Analyze AI/Tech sector
perf = analyzer.analyze_sector('AI/Tech')

print(f"Best Stock: {perf.best_stock} ({perf.best_stock_return:.2f}%)")
print(f"Best Strategy: {perf.best_strategy}")
print(f"Reliability: {perf.reliability_rating}")

# Compare all sectors
comparison = analyzer.compare_all_sectors()

for sector, return_val in comparison.by_return[:5]:
    details = comparison.sector_details[sector]
    print(f"{sector}: {return_val:.2f}% ({details.num_stocks} stocks)")
```

### Example Output

```
======================================================================
SECTOR DEEP DIVE: AI/Tech
======================================================================

OVERVIEW:
  Reliability:     High (Quality Score: 92.3/100)
  Stocks Tested:   15
  Strategies:      10
  Total Tests:     450

STRATIFIED PERFORMANCE (fixes large stdev problem!):

  CONSERVATIVE Strategies:
    Median Return:  8.50%
    Mean Return:    9.20% ± 1.15%
    Std Dev:        10.25% (MAD: 7.80%)
    95% CI:         [6.90%, 11.50%]
    Tests:          90

  MODERATE Strategies:
    Median Return:  14.30%
    Mean Return:    15.10% ± 1.85%
    Std Dev:        16.50% (MAD: 12.40%)
    95% CI:         [11.40%, 18.80%]
    Tests:          180

  AGGRESSIVE Strategies:
    Median Return:  22.70%
    Mean Return:    23.40% ± 2.50%
    Std Dev:        24.10% (MAD: 18.90%)
    95% CI:         [18.40%, 28.40%]
    Tests:          180

OVERALL STATISTICS:
  Median Return:   15.20%
  Mean Return:     16.80%
  Std Dev:         18.50%
  95% CI:          [14.10%, 19.50%]

RISK METRICS:
  Avg Sharpe Ratio:  1.45
  Avg Volatility:    19.80%
  Avg Win Rate:      68.5%

BEST PERFORMERS:
  Best Stock:      NVDA (+35.20%)
  Best Strategy:   Kalman Adaptive Filter (+24.50%)
  Strategy Consistency: 0.812

──────────────────────────────────────────────────────────────────────
TOP STOCKS IN AI/Tech:
──────────────────────────────────────────────────────────────────────

1. NVDA (Rank 1/15)
   Mean Return:    35.20% (Median: 33.50%)
   Best Strategy:  Kalman Adaptive Filter (+48.70%)
   Consistency:    0.856
   Tests:          30
   Reliability:    High

2. AMD (Rank 2/15)
   Mean Return:    28.90% (Median: 27.30%)
   Best Strategy:  Multi-Factor Scoring (+38.20%)
   Consistency:    0.792
   Tests:          30
   Reliability:    High

...
```

---

## Time Period Analysis

### Purpose

Understand how strategies perform across different **time horizons**.

### Python Usage

```python
from modules.analysis.period_analyzer import PeriodAnalyzer

# Load analyzer
analyzer = PeriodAnalyzer.from_csv('backtest_results/detailed_results_20251122_203908.csv')

# Analyze strategy across periods
profile = analyzer.analyze_strategy_by_period('Kalman Adaptive Filter')

print(f"Short-term (<6mo):   {profile.short_term_return:.2f}%")
print(f"Medium-term (6-12mo): {profile.medium_term_return:.2f}%")
print(f"Long-term (>1yr):    {profile.long_term_return:.2f}%")
print(f"Optimal Period:      {profile.optimal_period}")
print(f"Period Sensitive:    {profile.is_period_sensitive}")

# Find optimal period for each strategy
df = analyzer.find_optimal_period_by_strategy()
print(df)

# Analyze stock across periods
stock_profile = analyzer.analyze_stock_by_period('AAPL')
print(f"Best Period: {stock_profile.optimal_period}")
print(f"Best Strategy: {stock_profile.optimal_period_strategy}")
```

### Example Output

```
======================================================================
STRATEGY PERIOD ANALYSIS: Kalman Adaptive Filter
======================================================================

PERFORMANCE BY TIME HORIZON:
  Short-term (<6mo):      14.20%
  Medium-term (6-12mo):   18.50%
  Long-term (>1yr):       22.30%

OPTIMAL PERIOD:
  Period:              2yr
  Return:              24.80%

CONSISTENCY:
  Across Periods:      0.845 (1.0 = perfect consistency)
  Period Sensitive:    Yes - >10% variation

──────────────────────────────────────────────────────────────────────
DETAILED PERIOD BREAKDOWN:
──────────────────────────────────────────────────────────────────────

2yr (long, ~730d):
  Mean Return:       24.80%
  Annualized:        11.70%
  Sharpe Ratio:       1.82
  Success Rate:      92.5%
  Tests:             40
  Reliability:       High

1yr (medium, ~365d):
  Mean Return:       18.50%
  Annualized:        18.50%
  Sharpe Ratio:       1.45
  Success Rate:      87.5%
  Tests:             40
  Reliability:       High

...
```

---

## Advanced Filtering

### Purpose

**Multi-dimensional filtering** to drill down into specific scenarios.

### CLI Usage

```bash
# Filter by sector and minimum Sharpe
quantsploit > filter --sector AI/Tech --min-sharpe 1.0

# Filter by stock and return
quantsploit > filter --symbol AAPL --min-return 10

# Filter by strategy and win rate
quantsploit > filter --strategy "Kalman Adaptive Filter" --min-win-rate 60

# Get top 10 results
quantsploit > filter --min-sharpe 1.5 --top-n 10

# Combine multiple filters
quantsploit > filter --sector AI/Tech --min-return 15 --max-volatility 25 --top-n 5
```

### Python Usage

```python
from modules.analysis.advanced_filter import AdvancedFilter

# Load filter
filter_sys = AdvancedFilter.from_csv('backtest_results/detailed_results_20251122_203908.csv')

# Quick filter
results = filter_sys.quick_filter(
    sector='AI/Tech',
    min_sharpe=1.0,
    min_return=10,
    max_volatility=30,
    min_trades=5
)

print(f"Found {len(results)} results")

# Top N results
top_10 = filter_sys.top_n(
    n=10,
    sort_by='total_return',
    min_sharpe=1.2
)

# Rank by group
ranked = filter_sys.rank_by(
    group_by='symbol',
    metric='total_return',
    sector='AI/Tech',
    min_sharpe=1.0
)
```

---

## Complete Workflow Example

Here's how to execute your desired process flow:

### 1. Identify Sectors with Potential

```bash
quantsploit > analyze sector AI/Tech
quantsploit > analyze sector Semiconductors
quantsploit > analyze sector Cloud Computing
```

Or in Python:

```python
from modules.analysis.sector_deep_dive import SectorAnalyzer

analyzer = SectorAnalyzer.from_csv('backtest_results/detailed_results_20251122_203908.csv')
comparison = analyzer.compare_all_sectors()

# Top 5 sectors by return
for i, (sector, return_val) in enumerate(comparison.by_return[:5], 1):
    details = comparison.sector_details[sector]
    print(f"{i}. {sector}: {return_val:.2f}% ({details.reliability_rating})")

# Result: AI/Tech looks best with 16.8% median return, High reliability
```

### 2. Find Individual Stocks Worth Looking Into

```bash
# Filter stocks in best sector
quantsploit > filter --sector AI/Tech --min-return 15 --min-sharpe 1.2 --top-n 10
```

Or in Python:

```python
from modules.analysis.advanced_filter import AdvancedFilter

filter_sys = AdvancedFilter.from_csv('backtest_results/detailed_results_20251122_203908.csv')

# Get top stocks in AI/Tech with good metrics
top_stocks = filter_sys.rank_by(
    group_by='symbol',
    metric='total_return',
    sector='AI/Tech',
    min_sharpe=1.2,
    min_win_rate=65,
    top_n=10
)

print(top_stocks)

# Result: NVDA, AMD, MSFT, GOOGL show up consistently
```

### 3. Compare Strategies for Specific Stocks

```bash
# Deep dive on NVDA
quantsploit > analyze stock NVDA

# Compare top strategies on NVDA
quantsploit > compare "Kalman Adaptive Filter" "Multi-Factor Scoring" --stock NVDA
```

Result: Kalman beats Multi-Factor by 8.5% on NVDA with p < 0.01

### 4. Analyze Time Periods

```python
from modules.analysis.period_analyzer import PeriodAnalyzer

analyzer = PeriodAnalyzer.from_csv('backtest_results/detailed_results_20251122_203908.csv')

# How does Kalman perform across periods for NVDA?
stock_profile = analyzer.analyze_stock_by_period('NVDA')
print(f"Best Period: {stock_profile.optimal_period}")  # 1yr
print(f"Best Strategy: {stock_profile.optimal_period_strategy}")  # Kalman

# Is Kalman consistent across time?
strategy_profile = analyzer.analyze_strategy_by_period('Kalman Adaptive Filter')
print(f"Period Consistency: {strategy_profile.period_consistency}")  # 0.82 = good!
```

### 5. Make Final Decision

Based on analysis:

**✅ Stock: NVDA**
**✅ Strategy: Kalman Adaptive Filter**
**✅ Time Period: 1 year**
**✅ Expected Return: ~28% (95% CI: [22%, 34%])**
**✅ Sharpe Ratio: 1.85**
**✅ Win Rate: 72.5%**
**✅ Confidence: High (statistically significant, large sample)**

---

## CLI Commands Reference

### `analyze`

Analyze stocks, sectors, or periods.

```bash
analyze <stock|sector|period> <NAME> [--timestamp TS]
```

**Examples:**
```bash
analyze stock AAPL
analyze sector AI/Tech
analyze stock NVDA --timestamp 20251122_203908
```

### `compare`

Compare strategies head-to-head.

```bash
compare <STRATEGY1> <STRATEGY2> [--stock SYMBOL] [--sector SECTOR]
```

**Examples:**
```bash
compare "SMA Crossover (20/50)" "Kalman Adaptive Filter"
compare "SMA Crossover (20/50)" "Momentum (10/20/50)" --stock AAPL
compare "Multi-Factor Scoring" "HMM Regime Detection" --sector AI/Tech
```

### `filter`

Filter backtest results with multiple criteria.

```bash
filter [options]
```

**Options:**
- `--sector SECTOR` - Filter by sector
- `--symbol SYMBOL` - Filter by stock symbol
- `--strategy STRATEGY` - Filter by strategy
- `--period PERIOD` - Filter by time period
- `--min-return PCT` - Minimum return threshold
- `--min-sharpe RATIO` - Minimum Sharpe ratio
- `--min-win-rate PCT` - Minimum win rate
- `--max-volatility PCT` - Maximum volatility
- `--top-n N` - Show only top N results

**Examples:**
```bash
filter --sector AI/Tech --min-sharpe 1.0
filter --symbol AAPL --min-return 10
filter --min-sharpe 1.5 --top-n 10
filter --sector AI/Tech --min-return 15 --max-volatility 25
```

---

## Python API Reference

### StockAnalyzer

```python
from modules.analysis.stock_analyzer import StockAnalyzer

# Load from CSV or timestamp
analyzer = StockAnalyzer.from_csv(csv_path)
analyzer = StockAnalyzer.from_timestamp(timestamp)

# Analyze single stock
analysis = analyzer.analyze_stock('AAPL', min_trades=5)

# Get formatted summary
summary = analyzer.get_stock_summary('AAPL', min_trades=5)

# Compare multiple stocks
comparison = analyzer.compare_stocks(['AAPL', 'NVDA', 'MSFT'])
```

### StrategyComparator

```python
from modules.analysis.strategy_comparator import StrategyComparator

# Load
comparator = StrategyComparator.from_csv(csv_path)

# Compare two
result = comparator.compare_two_strategies(
    'Strategy1', 'Strategy2',
    stock='AAPL',  # Optional
    sector='AI/Tech'  # Optional
)

# Compare multiple
multi_result = comparator.compare_multiple_strategies(
    ['Strategy1', 'Strategy2', 'Strategy3'],
    stock='NVDA'
)

# Find best strategy per stock
df = comparator.compare_strategies_by_stock(['Strategy1', 'Strategy2'])
```

### SectorAnalyzer

```python
from modules.analysis.sector_deep_dive import SectorAnalyzer

# Load
analyzer = SectorAnalyzer.from_csv(csv_path)

# Analyze sector
perf = analyzer.analyze_sector('AI/Tech', min_trades=5)

# Compare all sectors
comparison = analyzer.compare_all_sectors()

# Format reports
report = analyzer.format_sector_report(perf)
comp_report = analyzer.format_sector_comparison(comparison)
```

### PeriodAnalyzer

```python
from modules.analysis.period_analyzer import PeriodAnalyzer

# Load
analyzer = PeriodAnalyzer.from_csv(csv_path)

# Analyze strategy by period
profile = analyzer.analyze_strategy_by_period('Strategy Name')

# Analyze stock by period
stock_profile = analyzer.analyze_stock_by_period('AAPL')

# Compare all periods
periods = analyzer.compare_periods(min_trades=5)

# Find optimal period per strategy
df = analyzer.find_optimal_period_by_strategy()
```

### AdvancedFilter

```python
from modules.analysis.advanced_filter import AdvancedFilter

# Load
filter_sys = AdvancedFilter.from_csv(csv_path)

# Quick filter
results = filter_sys.quick_filter(
    sector='AI/Tech',
    strategy='Kalman Adaptive Filter',
    min_return=10,
    min_sharpe=1.0,
    max_volatility=30,
    min_trades=5
)

# Top N
top_10 = filter_sys.top_n(
    n=10,
    sort_by='total_return',
    min_sharpe=1.2
)

# Rank by group
ranked = filter_sys.rank_by(
    group_by='symbol',
    metric='total_return',
    sector='AI/Tech'
)
```

---

## Key Features Summary

### ✅ Problems Solved

1. **Large Standard Deviation Issue** → Fixed with stratified statistics
2. **Can't analyze specific stocks** → Stock Deep Dive Analyzer
3. **Can't compare strategies** → Strategy Comparison Engine
4. **Can't see sector performance** → Enhanced Sector Analysis
5. **Can't analyze time periods** → Period Analyzer
6. **Can't drill down with filters** → Advanced Filtering System

### ✅ New Capabilities

- **Stratified Statistics** - Separate analysis by risk class
- **Robust Metrics** - Median, MAD, IQR instead of naive mean/stdev
- **Confidence Intervals** - Know the uncertainty in your estimates
- **Statistical Significance** - Mann-Whitney U tests for comparisons
- **Quality Scores** - Understand data reliability
- **Multi-dimensional Filtering** - Drill down with complex criteria

### ✅ Complete Workflow Support

Now you can execute the full analysis pipeline:
1. ✅ Identify promising sectors
2. ✅ Find top stocks in those sectors
3. ✅ Compare strategies for specific stocks
4. ✅ Analyze optimal time periods
5. ✅ Make confident, data-driven decisions

---

## Tips & Best Practices

1. **Always run with enough data** - Aim for at least 10+ stocks, 5+ periods, 5+ strategies
2. **Check reliability ratings** - Only trust "High" or "Medium" reliability results
3. **Use stratified statistics** - Don't rely on overall means when comparing diverse strategies
4. **Filter wisely** - Start broad, then narrow down with multiple filters
5. **Compare apples to apples** - When comparing strategies, use the same stock/period filters
6. **Consider time horizons** - A strategy that works for 1yr may not work for 6mo
7. **Look at consistency** - High average return with low consistency = risky
8. **Use confidence intervals** - Understand the range of possible outcomes

---

## Troubleshooting

**Q: I get "No backtest results found"**
A: Run a comprehensive backtest first: `use backtesting/comprehensive` then `run`

**Q: My analysis shows "Low reliability"**
A: You need more data. Run backtests with more stocks, periods, or strategies

**Q: The compare command shows "Insufficient data"**
A: Make sure both strategies have been tested on the same stocks/periods

**Q: Large standard deviations still appearing**
A: Use the stratified statistics view or filter by specific risk class

**Q: Import errors when running commands**
A: Make sure you're in the Quantsploit root directory and modules are properly installed

---

## Future Enhancements

- Dashboard integration for visual analysis
- PDF report generation
- Portfolio optimization across multiple stocks
- Machine learning for strategy selection
- Real-time monitoring and alerts
- Regime detection and strategy switching

---

## Questions?

Check the main Quantsploit documentation or create an issue on GitHub.

**Happy Trading! 📈**
