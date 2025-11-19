# Advanced Quantitative Strategies Guide

## Overview

This guide covers the advanced quantitative analysis modules in Quantsploit, designed to analyze large numbers of tickers, detect patterns, generate buy/sell signals, and identify top trading opportunities.

## 🚀 New Modules

### 1. Advanced Bulk Screener (`scanners/bulk_screener`)

High-performance parallel screening of large stock universes.

**Features:**
- Parallel processing (10+ workers)
- Pre-defined lists: SP500, NASDAQ100, DOW30
- Advanced filters: price, volume, RSI, trend
- Multiple sorting options

**Usage:**
```
quantsploit > use scanners/bulk_screener
quantsploit (Advanced Bulk Screener) > set SYMBOLS SP500
quantsploit (Advanced Bulk Screener) > set MIN_VOLUME 1000000
quantsploit (Advanced Bulk Screener) > set RSI_MIN 30
quantsploit (Advanced Bulk Screener) > set RSI_MAX 70
quantsploit (Advanced Bulk Screener) > set SORT_BY score
quantsploit (Advanced Bulk Screener) > run
```

**Options:**
- `SYMBOLS`: SP500, NASDAQ100, DOW30, or comma-separated list
- `MIN_PRICE`: Minimum stock price (default: 5.0)
- `MAX_PRICE`: Maximum stock price (default: 10000)
- `MIN_VOLUME`: Minimum daily volume (default: 500000)
- `RSI_MIN/MAX`: RSI range filter
- `TREND_FILTER`: uptrend, downtrend, or None
- `SORT_BY`: score, volume, price_change, rsi, momentum
- `MAX_WORKERS`: Parallel workers (default: 10)

### 2. Pattern Recognition (`analysis/pattern_recognition`)

Detects candlestick and chart patterns with buy/sell signals.

**Features:**
- Candlestick patterns: Hammer, Shooting Star, Engulfing, Doji, Morning/Evening Star
- Chart patterns: Double Bottom, Head & Shoulders, Triangles
- Support/Resistance levels
- Automated signal generation

**Usage:**
```
quantsploit > use analysis/pattern_recognition
quantsploit (Pattern Recognition) > set SYMBOL AAPL
quantsploit (Pattern Recognition) > set LOOKBACK 50
quantsploit (Pattern Recognition) > run
```

**Detected Patterns:**
- **Bullish:** Hammer, Bullish Engulfing, Morning Star, Double Bottom, Ascending Triangle
- **Bearish:** Shooting Star, Bearish Engulfing, Evening Star, Head & Shoulders
- **Neutral:** Doji (indecision)

### 3. Mean Reversion Strategy (`strategies/mean_reversion`)

Statistical mean reversion with z-score analysis.

**Features:**
- Z-score calculation
- Bollinger Bands analysis
- Percentile ranking
- Mean reversion probability
- Expected return to mean

**Usage:**
```
quantsploit > use strategies/mean_reversion
quantsploit (Mean Reversion Strategy) > set SYMBOL TSLA
quantsploit (Mean Reversion Strategy) > set LOOKBACK 20
quantsploit (Mean Reversion Strategy) > set Z_THRESHOLD 2.0
quantsploit (Mean Reversion Strategy) > run
```

**Signals:**
- Z-score < -2.0: Strong oversold (BUY)
- Z-score > 2.0: Strong overbought (SELL)
- Below lower Bollinger Band: BUY signal
- Above upper Bollinger Band: SELL signal

### 4. Momentum Signals (`strategies/momentum_signals`)

Advanced momentum and trend following strategy.

**Features:**
- Multi-period Rate of Change (ROC)
- ADX trend strength
- Volume-weighted momentum
- Momentum acceleration
- Relative strength vs benchmark
- Moving average alignment

**Usage:**
```
quantsploit > use strategies/momentum_signals
quantsploit (Momentum Signals) > set SYMBOL NVDA
quantsploit (Momentum Signals) > set BENCHMARK SPY
quantsploit (Momentum Signals) > set MIN_ADX 25
quantsploit (Momentum Signals) > run
```

**Scoring Components:**
- 12-period momentum > 10%: +25 points
- Momentum acceleration: +20 points
- Strong trend (ADX > 25): +15 points
- Above VWAP: +10 points
- MA alignment: +20 points
- Outperforming benchmark: +15 points

### 5. Multi-Factor Scoring (`strategies/multifactor_scoring`)

Comprehensive quantitative scoring system combining multiple factors.

**Features:**
- Momentum factors
- Technical factors (RSI, MACD, MA)
- Volatility factors (ATR, std dev)
- Volume factors (OBV, volume trend)
- Composite scoring (0-100)
- Customizable factor weights

**Usage:**
```
quantsploit > use strategies/multifactor_scoring
quantsploit (Multi-Factor Scoring) > set SYMBOLS AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA
quantsploit (Multi-Factor Scoring) > set FACTOR_WEIGHTS momentum:0.3,technical:0.3,volatility:0.2,volume:0.2
quantsploit (Multi-Factor Scoring) > run
```

**Score Interpretation:**
- 75-100: STRONG BUY
- 60-75: BUY
- 40-60: HOLD
- 25-40: SELL
- 0-25: STRONG SELL

### 6. Signal Aggregator (`analysis/signal_aggregator`)

Combines multiple strategies for consensus buy/sell signals.

**Features:**
- Aggregates 5 different strategies
- Confidence scoring
- Risk assessment
- Actionable insights
- Conflict detection

**Usage:**
```
quantsploit > use analysis/signal_aggregator
quantsploit (Signal Aggregator) > set SYMBOL AAPL
quantsploit (Signal Aggregator) > set MIN_CONFIDENCE 60
quantsploit (Signal Aggregator) > run
```

**Aggregated Strategies:**
1. Momentum analysis
2. Mean reversion analysis
3. Technical indicators
4. Pattern recognition
5. Volume analysis

**Output:**
- Final signal with confidence %
- Individual strategy signals
- Risk assessment (HIGH/MEDIUM/LOW)
- Actionable insights

### 7. Meta-Analysis (`analysis/meta_analysis`)

**NEW!** Advanced meta-analysis that runs ALL strategies and correlates signals to find stocks with the most consistent signaling across different approaches.

**Features:**
- Runs 8+ trading strategies simultaneously
- Normalizes signals across different strategy types
- Calculates weighted consensus (strength & confidence)
- Ranks stocks by signal consistency
- Multi-symbol analysis support
- Detailed strategy breakdown

**Usage:**
```
quantsploit > use analysis/meta_analysis
quantsploit (Meta-Analysis) > set SYMBOLS AAPL,MSFT,GOOGL,NVDA,TSLA
quantsploit (Meta-Analysis) > set MIN_CONSENSUS 60
quantsploit (Meta-Analysis) > set PERIOD 1y
quantsploit (Meta-Analysis) > run
```

**Options:**
- `SYMBOLS`: Comma-separated list of stocks to analyze (REQUIRED)
- `PERIOD`: Historical data period (default: 1y)
- `INTERVAL`: Data interval (default: 1d)
- `MIN_CONSENSUS`: Minimum agreement % to highlight (default: 60)
- `STRATEGIES`: Specific strategies to run, or 'all' (default: all)

**Strategies Analyzed:**
1. SMA Crossover
2. Mean Reversion
3. Momentum Signals
4. ML Swing Trading
5. Multi-Factor Scoring
6. Volume Profile Swing
7. Kalman Adaptive
8. HMM Regime Detection

**Output:**
- Ranked list of symbols by consensus strength
- Consensus signal (BUY/SELL/HOLD)
- Agreement percentage across strategies
- Buy/Sell/Hold counts
- Detailed breakdown of each strategy's signal
- Visual indicators for high-confidence signals

**Interpretation:**
- 🟢 Strong BUY: High agreement on bullish signals (≥ MIN_CONSENSUS)
- 🔴 Strong SELL: High agreement on bearish signals (≥ MIN_CONSENSUS)
- ⚪ Mixed/HOLD: Low consensus or neutral signals

**Use Cases:**
- Find stocks with the highest signal consensus
- Identify which strategies agree/disagree for a symbol
- Validate high-conviction trades across multiple methodologies
- Compare signal strength across a portfolio of stocks
- Discover the most "tradeable" stocks with clear signals

### 8. Top Movers & Rankings (`scanners/top_movers`)

Identifies and ranks top opportunities across multiple dimensions.

**Features:**
- Top gainers/losers
- Momentum leaders
- Near breakout stocks
- Oversold opportunities
- High-quality stocks
- Volume leaders

**Usage:**
```
quantsploit > use scanners/top_movers
quantsploit (Top Movers & Rankings) > set SYMBOLS SP500
quantsploit (Top Movers & Rankings) > set RANKING_METHOD all
quantsploit (Top Movers & Rankings) > set TIMEFRAME 1mo
quantsploit (Top Movers & Rankings) > set TOP_N 20
quantsploit (Top Movers & Rankings) > run
```

**Ranking Methods:**
- `all`: All rankings
- `gainers`: Top % gainers
- `momentum`: Best momentum scores
- `breakout`: Near 52-week highs
- `oversold`: Best oversold opportunities
- `quality`: Highest quality scores
- `volume`: High volume stocks

## 📊 Complete Workflow Examples

### Example 1: Finding Top Momentum Plays

```bash
# Step 1: Screen large universe
use scanners/bulk_screener
set SYMBOLS SP500
set TREND_FILTER uptrend
set MIN_VOLUME 2000000
set SORT_BY momentum
run

# Step 2: Analyze top candidates with momentum strategy
use strategies/momentum_signals
set SYMBOL NVDA
set BENCHMARK SPY
run

# Step 3: Get consensus signal
use analysis/signal_aggregator
set SYMBOL NVDA
run
```

### Example 2: Finding Oversold Reversal Opportunities

```bash
# Step 1: Screen for oversold stocks
use scanners/bulk_screener
set SYMBOLS NASDAQ100
set RSI_MAX 35
set SORT_BY rsi
run

# Step 2: Analyze mean reversion potential
use strategies/mean_reversion
set SYMBOL AMD
set Z_THRESHOLD 2.0
run

# Step 3: Check for bullish patterns
use analysis/pattern_recognition
set SYMBOL AMD
run
```

### Example 3: Multi-Factor Stock Ranking

```bash
# Rank and score multiple stocks
use strategies/multifactor_scoring
set SYMBOLS AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,AMD,NFLX,ADBE
set FACTOR_WEIGHTS momentum:0.35,technical:0.30,volatility:0.20,volume:0.15
run
```

### Example 4: Daily Top Movers Analysis

```bash
# Find today's top movers and opportunities
use scanners/top_movers
set SYMBOLS SP500
set RANKING_METHOD all
set TIMEFRAME 1d
set TOP_N 15
run
```

### Example 5: Meta-Analysis for High-Conviction Trades

```bash
# Run all strategies on multiple stocks to find consensus
use analysis/meta_analysis
set SYMBOLS AAPL,MSFT,GOOGL,NVDA,TSLA,AMD,META,NFLX
set MIN_CONSENSUS 70
set PERIOD 1y
run

# This will:
# 1. Run 8+ strategies on each symbol
# 2. Show which stocks have the highest signal consensus
# 3. Rank stocks by signal strength
# 4. Display detailed breakdown of each strategy's signal
```

## 🎯 Strategy Selection Guide

### When to Use Each Strategy

**Bulk Screener** → Initial filtering of large universes
- Use when you need to narrow down hundreds of stocks
- Best for finding candidates that meet specific criteria

**Pattern Recognition** → Entry/exit timing
- Use when you want to identify technical setups
- Best for timing entries around support/resistance

**Mean Reversion** → Oversold/overbought plays
- Use when looking for reversal opportunities
- Best in range-bound or sideways markets

**Momentum Signals** → Trend following
- Use when looking for continuation plays
- Best in strong trending markets

**Multi-Factor Scoring** → Comprehensive ranking
- Use when comparing multiple stocks
- Best for portfolio construction

**Signal Aggregator** → Final decision making
- Use before entering a trade
- Best for confirming high-conviction setups

**Meta-Analysis** → Multi-stock consensus validation
- Use when analyzing multiple stocks to find the best opportunities
- Best for finding stocks with the most consistent signals across ALL strategies
- Ideal for portfolio construction and high-conviction trade selection

**Top Movers** → Finding opportunities
- Use daily to scan for active stocks
- Best for discovering new ideas

## 🔥 Pro Tips

### 1. Parallel Workflow
Analyze multiple stocks simultaneously:
```bash
# Terminal 1
use strategies/momentum_signals
set SYMBOL AAPL
run

# Terminal 2
use strategies/mean_reversion
set SYMBOL AAPL
run

# Terminal 3
use analysis/signal_aggregator
set SYMBOL AAPL
run
```

### 2. Custom Watchlists
Build watchlists from scanner results:
```bash
use scanners/top_movers
set SYMBOLS SP500
set RANKING_METHOD momentum
run
# Note top symbols, then:
watchlist add NVDA Strong momentum leader
watchlist add AMD High quality score
watchlist show
```

### 3. Factor Weight Optimization
Test different factor weights:
```bash
# Growth focused
set FACTOR_WEIGHTS momentum:0.5,technical:0.3,volatility:0.1,volume:0.1

# Conservative
set FACTOR_WEIGHTS momentum:0.2,technical:0.2,volatility:0.4,volume:0.2

# Momentum + Volume
set FACTOR_WEIGHTS momentum:0.4,technical:0.2,volatility:0.1,volume:0.3
```

### 4. Risk Management
Always check:
1. Signal Aggregator for confidence level
2. Mean Reversion for volatility metrics
3. Pattern Recognition for support/resistance
4. Top Movers for relative performance

### 5. Meta-Analysis Workflow
For best results with meta-analysis:
```bash
# Step 1: Screen for candidates
use scanners/bulk_screener
set SYMBOLS SP500
set MIN_VOLUME 2000000
run

# Step 2: Run meta-analysis on top 10 candidates
use analysis/meta_analysis
set SYMBOLS AAPL,MSFT,GOOGL,NVDA,TSLA,AMD,META,AMZN,NFLX,ADBE
set MIN_CONSENSUS 65
run

# Step 3: Deep dive on stocks with highest consensus
use analysis/signal_aggregator
set SYMBOL NVDA  # The stock with highest consensus from meta-analysis
run
```

## ⚡ Performance Tips

- Use `MAX_WORKERS` parameter to speed up large scans
- Cache enabled by default (3600s)
- Run bulk screens during off-market hours
- Use appropriate PERIOD for your strategy:
  - Day trading: PERIOD=5d, INTERVAL=1m
  - Swing trading: PERIOD=3mo, INTERVAL=1d
  - Position trading: PERIOD=1y, INTERVAL=1d

## 📈 Signal Interpretation

### Strong Buy Signals (High Conviction)
- Signal Aggregator confidence > 80%
- **Meta-Analysis consensus > 70% with BUY signal**
- Multi-factor score > 75
- Multiple bullish patterns detected
- Momentum score > 60 with low risk

### Buy Signals (Moderate Conviction)
- Signal Aggregator confidence 60-80%
- **Meta-Analysis consensus 60-70% with BUY signal**
- Multi-factor score 60-75
- One or two bullish confirmations
- Oversold with reversal signs

### Hold/Neutral
- Mixed signals from aggregator
- Multi-factor score 40-60
- No clear pattern or trend
- Low confidence (<60%)

### Sell Signals
- Signal Aggregator bearish with >60% confidence
- Multi-factor score < 40
- Bearish patterns detected
- Overbought conditions

## 🛡️ Risk Disclaimer

These strategies are for educational purposes. Always:
- Use proper position sizing
- Set stop losses
- Diversify your portfolio
- Consider your risk tolerance
- Do your own research

## 📚 Further Reading

- See `README.md` for basic usage
- Check module source code for detailed algorithms
- Review `config.yaml` for customization options

---

**Built with cutting-edge quantitative techniques. Trade smart, not hard! 📊**
