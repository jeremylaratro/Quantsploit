# Reddit Sentiment Analysis System

A comprehensive sentiment analysis system that spiders Reddit for stock ticker mentions and analyzes sentiment based on verbiage and word choice. Includes a real-time dashboard and trading strategy.

## Overview

This feature adds three main components to Quantsploit:

1. **Reddit Sentiment Analyzer** (`quantsploit/modules/analysis/reddit_sentiment.py`) - Spiders Reddit for ticker mentions and analyzes sentiment
2. **Reddit Sentiment Trading Strategy** (`quantsploit/modules/strategies/reddit_sentiment_strategy.py`) - Generates trading signals based on sentiment
3. **Web Dashboard** (`dashboard/templates/reddit_sentiment.html`) - Interactive visualization of sentiment data

## Features

### Sentiment Analyzer
- **Multi-Subreddit Analysis**: Simultaneously analyzes multiple subreddits (wallstreetbets, stocks, investing, etc.)
- **Ticker Extraction**: Automatically extracts stock ticker symbols from post titles and content
- **Contextual Sentiment**: VADER sentiment with finance-specific lexical cues and ticker-aware sentence context (titles weighted higher)
- **Comment Analysis**: Optional deep analysis of comments for comprehensive sentiment
- **Quality Filtering**: Filters by upvotes, post scores, and mention frequency
- **Sentiment Categories**: Classifies sentiment as Very Bullish, Bullish, Neutral, Bearish, Very Bearish

### Trading Strategy
- **Signal Generation**: BUY/HOLD/SELL signals based on sentiment thresholds
- **Price Confirmation**: Optional price momentum confirmation to filter false signals
- **Risk Management**: Built-in stop loss, take profit, and position sizing
- **Multi-Factor Scoring**: Combines sentiment score, mention volume, post quality, and sentiment distribution
- **Backtest Ready**: Fully integrated with Quantsploit's backtesting engine

### Dashboard
- **Real-Time Analysis**: Run sentiment analysis on-demand from the web interface
- **Interactive Charts**: Sentiment distribution pie chart and scatter plots
- **Top Tickers Table**: Ranked by mentions and sentiment strength
- **Trading Signals**: Automatic signal generation with confidence levels
- **Post Browser**: View top Reddit posts for each ticker
- **Customizable Parameters**: Adjust subreddits, time filters, and thresholds

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   New dependencies added:
   - `praw` - Python Reddit API Wrapper
   - `nltk` - Natural Language Toolkit
   - `vaderSentiment` - Sentiment analysis specifically tuned for social media
   - `textblob` - Additional NLP capabilities

2. **Reddit API Credentials (Required by Reddit)**

   Reddit now requires real API credentials even for read-only access. If you skip this step you will see 401 errors.

   a. Create a Reddit app at https://www.reddit.com/prefs/apps  
      - Click "create another app" -> type: **script**  
      - Name: e.g. "Quantsploit Sentiment"  
      - Redirect URI: `http://localhost:8080` (unused here but required)
   b. Export credentials before running the analyzer (use your Reddit username in the user agent):
      ```bash
      export REDDIT_CLIENT_ID="your_client_id"
      export REDDIT_CLIENT_SECRET="your_client_secret"
      export REDDIT_USER_AGENT="Quantsploit Sentiment (by u/your_username)"
      ```
      Windows (Powershell):
      ```powershell
      $env:REDDIT_CLIENT_ID="your_client_id"
      $env:REDDIT_CLIENT_SECRET="your_client_secret"
      $env:REDDIT_USER_AGENT="Quantsploit Sentiment (by u/your_username)"
      ```
   c. If you see HTTP 401: regenerate the secret, confirm the app type is **script**, and double-check the env vars are set in the shell that launches Quantsploit.

3. **API-Free Scrape Mode**

   If you cannot request API access, set `ACCESS_MODE` to `scrape` to use the public JSON endpoints on `old.reddit.com` (no credentials). Comment analysis is disabled in this mode.
   ```bash
   export ACCESS_MODE="scrape"
   export REDDIT_USER_AGENT="Quantsploit Reddit Sentiment (scrape mode)"
   ```

## Usage

Make sure `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT` are set in your shell before starting the CLI or dashboard.

### Command Line Interface

1. **Basic Sentiment Analysis**:
   ```bash
   python -m quantsploit.main
   use analysis/reddit_sentiment
   set SUBREDDITS wallstreetbets,stocks,investing
   set TIME_FILTER day
   set POST_LIMIT 100
   run
   ```

2. **Analyze Specific Ticker**:
   ```bash
   use analysis/reddit_sentiment
   set FILTER_SYMBOL TSLA
   set MIN_MENTIONS 5
   run
   ```

3. **Run Trading Strategy**:
   ```bash
   use strategies/reddit_sentiment_strategy
   set SYMBOL AAPL
   set SENTIMENT_ENTRY_THRESHOLD 0.3
   set USE_PRICE_CONFIRMATION true
   run
   ```

### Web Dashboard

1. **Start the Dashboard**:
   ```bash
   cd dashboard
   python app.py
   ```

2. **Access Sentiment Analysis**:
   - Open browser to `http://localhost:5000`
   - Click "Reddit Sentiment" in navigation
   - Dashboard defaults to `ACCESS_MODE=scrape` (old.reddit.com JSON, no comments) so it works without API creds
   - Configure analysis parameters:
     - Subreddits to analyze
     - Time filter (hour, day, week, month)
     - Number of posts to fetch
   - Click "Run Analysis"

3. **View Results**:
   - Top 10 most mentioned tickers
   - Sentiment distribution charts
   - Trading signals with confidence levels
   - Individual ticker deep-dive with top posts

## Configuration Options

### Sentiment Analyzer Options

| Option | Default | Description |
|--------|---------|-------------|
| SUBREDDITS | wallstreetbets,stocks,investing,StockMarket,options | Comma-separated list of subreddits |
| SORT | top | Sort order: top, new, hot |
| TIME_FILTER | day | For SORT=top: hour, day, week, month, year, all (ignored for other sorts) |
| POST_LIMIT | 100 | Number of posts per subreddit |
| MIN_SCORE | 10 | Minimum upvote score for posts |
| ANALYZE_COMMENTS | True | Whether to analyze comments |
| COMMENT_LIMIT | 50 | Max comments per post |
| FILTER_SYMBOL | "" | Only analyze specific ticker |
| MIN_MENTIONS | 3 | Minimum mentions to include ticker |
| ACCESS_MODE | auto | auto uses API if creds exist, else scrape; api forces PRAW; scrape uses public JSON (no comments) |

### Trading Strategy Options

| Option | Default | Description |
|--------|---------|-------------|
| SENTIMENT_ENTRY_THRESHOLD | 0.3 | Minimum sentiment to enter (0-1) |
| SENTIMENT_EXIT_THRESHOLD | 0.1 | Sentiment below which to exit |
| MIN_POST_SCORE | 20 | Minimum average post quality |
| MAX_NEGATIVE_RATIO | 0.3 | Max ratio of negative mentions |
| STOP_LOSS_PCT | 5.0 | Stop loss percentage |
| TAKE_PROFIT_PCT | 15.0 | Take profit percentage |
| MAX_HOLD_DAYS | 5 | Maximum holding period |
| USE_PRICE_CONFIRMATION | True | Require price momentum confirmation |

## How It Works

### Sentiment Analysis Pipeline

1. **Data Collection**:
   - Connects to Reddit API
   - Fetches hot/top posts from specified subreddits
   - Extracts post titles, content, and comments

2. **Ticker Extraction**:
   - Regex pattern matching for uppercase tickers (1-5 letters)
   - Filters common words (AND, OR, THE, etc.)
   - Validates against known stock symbols

3. **Sentiment Scoring**:
   - VADER analyzes each mention
   - Compound score: -1 (very negative) to +1 (very positive)
   - Aggregates sentiment across all mentions

4. **Signal Generation**:
   - Calculates weighted sentiment strength
   - Considers mention volume, post quality, sentiment distribution
   - Generates BUY/HOLD/SELL signals with confidence levels

### Sentiment Categories

| Score Range | Category | Signal |
|-------------|----------|--------|
| ≥ 0.5 | Very Bullish | Strong BUY |
| 0.25 to 0.5 | Bullish | BUY |
| 0.05 to 0.25 | Slightly Bullish | HOLD |
| -0.05 to 0.05 | Neutral | HOLD |
| -0.25 to -0.05 | Slightly Bearish | SELL |
| -0.5 to -0.25 | Bearish | SELL |
| < -0.5 | Very Bearish | Strong SELL |

## Example Output

```
================================================================================
Reddit Sentiment Analysis Results
================================================================================
Subreddits: wallstreetbets, stocks, investing
Time Filter: day
Posts Analyzed: 287
Comments Analyzed: 4,325
Tickers Found: 45
================================================================================

Top 10 Most Mentioned Tickers with Sentiment:

Ticker   Mentions   Sentiment    Category           Strength
------------------------------------------------------------------------
TSLA     156        +0.4521      Bullish            42.3891
AAPL     89         +0.2134      Slightly Bullish   18.7654
SPY      67         +0.1234      Slightly Bullish   12.4567
NVDA     54         +0.6789      Very Bullish       29.8765
AMD      45         -0.1234      Slightly Bearish   5.4321
```

## Trading Strategy Example

```
🟢 BUY SIGNAL: NVDA has Very Bullish sentiment (score: 0.679) with 54 mentions.
Signal strength: 87/100.
Reasons: Very bullish sentiment, High mention volume (54), High quality discussions, Very low negative sentiment

Price Confirmation: ✓ CONFIRMED
- Strong 5-day momentum (+8.3%)
- Volume surge (+127%)
- Momentum score: 85/100

Recommendation: STRONG BUY - Enter position with 10% of capital
```

## Best Practices

### For Analysis
1. **Multiple Subreddits**: Analyze several communities for consensus
2. **Time Filters**: Use "day" or "week" for current sentiment, "hour" for very short-term
3. **Quality Filtering**: Set MIN_SCORE to filter low-quality posts
4. **Comment Analysis**: Enable for more comprehensive sentiment (slower but more accurate)

### For Trading
1. **Price Confirmation**: Always enable to filter false signals
2. **Avoid Meme Stocks**: Very high mentions with extreme sentiment often indicate pump schemes
3. **Risk Management**: Use stop losses - Reddit sentiment can reverse quickly
4. **Combine with Technicals**: Use sentiment as confirmation, not sole signal
5. **Short Hold Periods**: Social sentiment is often short-lived (3-7 days max)

### Warnings
- **Manipulation Risk**: Reddit communities can be manipulated
- **Volatility**: Sentiment-driven moves are highly volatile
- **Lagging Indicator**: By the time something trends on Reddit, the move may be over
- **False Positives**: Not all highly-mentioned tickers make good trades
- **Liquidity**: Stick to liquid, large-cap stocks

## Advanced Usage

### Custom Sentiment Thresholds

For more conservative trading:
```bash
set SENTIMENT_ENTRY_THRESHOLD 0.5  # Only very bullish
set MIN_MENTIONS 20                # High volume required
set MIN_POST_SCORE 50              # High quality only
```

For aggressive/momentum trading:
```bash
set SENTIMENT_ENTRY_THRESHOLD 0.2  # Lower threshold
set USE_PRICE_CONFIRMATION true    # But require price momentum
set MAX_HOLD_DAYS 3                # Quick in-and-out
```

### Backtesting the Strategy

```bash
use strategies/reddit_sentiment_strategy
set PERIOD 3mo
set INTERVAL 1d
set INITIAL_CAPITAL 10000
run
```

Note: Historical Reddit data is not available through this module. For backtesting, you would need historical sentiment data.

## Architecture

### Module Structure
```
quantsploit/
├── modules/
│   ├── analysis/
│   │   └── reddit_sentiment.py      # Sentiment analyzer
│   └── strategies/
│       └── reddit_sentiment_strategy.py  # Trading strategy
└── utils/
    └── data_fetcher.py              # Extended for sentiment caching

dashboard/
├── templates/
│   └── reddit_sentiment.html        # Web dashboard
└── app.py                           # Flask routes
```

### Data Flow
1. User triggers analysis (CLI or web)
2. RedditSentiment module fetches data from Reddit API (or scrape mode if API access is unavailable)
3. VADER analyzes sentiment of each mention
4. Results aggregated and cached
5. Trading strategy generates signals based on sentiment
6. Dashboard visualizes results

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'praw'**
   ```bash
   pip install praw vaderSentiment nltk textblob
   ```

2. **Reddit API Rate Limiting**
   - Solution: Reduce POST_LIMIT or use Reddit API credentials

3. **No Results Returned**
   - Check if subreddit names are correct
   - Reduce MIN_MENTIONS threshold
   - Increase TIME_FILTER to "week" or "month"

4. **Incorrect Ticker Extraction**
   - Common words filtered by EXCLUDE_WORDS set
   - Add custom exclusions to the EXCLUDE_WORDS in reddit_sentiment.py

## Future Enhancements

Potential improvements:
- [ ] Historical sentiment data storage for backtesting
- [ ] Twitter/X integration
- [ ] LLM-based sentiment analysis (GPT-4 for context understanding)
- [ ] Sentiment momentum tracking (rate of change)
- [ ] Cross-platform sentiment aggregation
- [ ] Sentiment divergence alerts (Reddit vs price action)
- [ ] Whale/influencer tracking
- [ ] Automated trading execution based on sentiment

## Contributing

To add new sentiment sources:
1. Create new analyzer in `quantsploit/modules/analysis/`
2. Follow BaseModule pattern
3. Return results in standard format
4. Add API endpoint to dashboard/app.py
5. Create visualization template

## License

Part of the Quantsploit project. See main LICENSE file.

## Credits

- VADER Sentiment: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
- PRAW: Python Reddit API Wrapper
- Quantsploit Framework: Jeremy Laratro
