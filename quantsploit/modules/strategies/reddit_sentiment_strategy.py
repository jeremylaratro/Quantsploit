"""
Reddit Sentiment Trading Strategy
Trade based on social media sentiment from Reddit
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta
from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher
from quantsploit.modules.analysis.reddit_sentiment import RedditSentiment


class RedditSentimentStrategy(BaseModule):
    """
    Trading strategy based on Reddit sentiment analysis.

    Entry Rules:
    - BUY when sentiment score > threshold AND mentions > min_mentions
    - Additional confirmation from sentiment momentum (increasing positive sentiment)

    Exit Rules:
    - SELL when sentiment drops below exit threshold
    - SELL when sentiment reverses (positive to negative)
    - Stop loss and take profit levels
    """

    @property
    def name(self) -> str:
        return "Reddit Sentiment Strategy"

    @property
    def description(self) -> str:
        return "Trade based on Reddit sentiment analysis and social media buzz"

    def trading_guide(self) -> str:
        return """SYNOPSIS: Trades stocks based on sentiment analysis from Reddit (r/wallstreetbets,
r/stocks, r/investing, etc.). Enters positions when strong positive sentiment is detected
with high mention volume.

SIMULATION POSITIONS:
  - BUY when sentiment score > entry threshold (default 0.3) AND mentions > min threshold
  - SELL when sentiment drops below exit threshold or reverses
  - Position sizing based on sentiment strength
  - Risk management with stop loss and take profit

RECOMMENDED ENTRY:
  - Sentiment score > 0.3 (bullish): Enter long position
  - High mention volume (>10 mentions): Confirms social media buzz
  - Increasing sentiment momentum: Additional confirmation
  - Avoid tickers with mixed sentiment (high positive AND negative mentions)

KEY SIGNALS:
  - Very Bullish sentiment (>0.5): Strong buy signal
  - Increasing mentions over time: Growing interest
  - High average post score: Quality discussions
  - Positive sentiment momentum: Sentiment improving
  - Cross-subreddit consensus: Multiple communities bullish

EXIT RULES:
  - Sentiment drops below exit threshold (default 0.1)
  - Sentiment reverses (positive to negative)
  - Stop loss hit (-5% default)
  - Take profit hit (+15% default)
  - Position held for max hold period (5 days default)

BEST USE:
  - Capture social media-driven momentum
  - Early detection of trending stocks
  - Combine with technical analysis for confirmation
  - Monitor for pump-and-dump patterns (very high mentions, extreme sentiment)

WARNINGS:
  - Social media sentiment can be manipulated
  - High volatility during sentiment spikes
  - Works best with liquid stocks (avoid penny stocks)
  - Reddit sentiment is often contrarian indicator for meme stocks
"""

    def show_info(self):
        info = super().show_info()
        info['trading_guide'] = self.trading_guide()
        return info

    @property
    def author(self) -> str:
        return "Quantsploit Team"

    @property
    def category(self) -> str:
        return "strategy"

    def _init_options(self):
        super()._init_options()
        self.options.update({
            "SUBREDDITS": {
                "value": "wallstreetbets,stocks,investing,StockMarket",
                "required": False,
                "description": "Comma-separated list of subreddits to analyze"
            },
            "TIME_FILTER": {
                "value": "day",
                "required": False,
                "description": "Time filter: hour, day, week, month"
            },
            "POST_LIMIT": {
                "value": 100,
                "required": False,
                "description": "Number of posts to fetch per subreddit"
            },
            "MIN_MENTIONS": {
                "value": 5,
                "required": False,
                "description": "Minimum mentions required to consider ticker"
            },
            "SENTIMENT_ENTRY_THRESHOLD": {
                "value": 0.3,
                "required": False,
                "description": "Minimum sentiment score to enter position (0-1)"
            },
            "SENTIMENT_EXIT_THRESHOLD": {
                "value": 0.1,
                "required": False,
                "description": "Sentiment score below which to exit position"
            },
            "MIN_POST_SCORE": {
                "value": 20,
                "required": False,
                "description": "Minimum average post score (quality filter)"
            },
            "MAX_NEGATIVE_RATIO": {
                "value": 0.3,
                "required": False,
                "description": "Max ratio of negative mentions (0-1)"
            },
            "STOP_LOSS_PCT": {
                "value": 5.0,
                "required": False,
                "description": "Stop loss percentage"
            },
            "TAKE_PROFIT_PCT": {
                "value": 15.0,
                "required": False,
                "description": "Take profit percentage"
            },
            "MAX_HOLD_DAYS": {
                "value": 5,
                "required": False,
                "description": "Maximum days to hold position"
            },
            "POSITION_SIZE_PCT": {
                "value": 10.0,
                "required": False,
                "description": "Position size as % of capital"
            },
            "USE_PRICE_CONFIRMATION": {
                "value": True,
                "required": False,
                "description": "Require price momentum confirmation"
            },
            "INITIAL_CAPITAL": {
                "value": 10000,
                "required": False,
                "description": "Initial capital for backtesting"
            }
        })

    def run(self) -> Dict[str, Any]:
        """Execute Reddit sentiment strategy analysis"""

        # Get sentiment data
        sentiment_module = RedditSentiment(self.framework)

        # Copy options to sentiment module
        sentiment_module.set_option("SUBREDDITS", self.get_option("SUBREDDITS"))
        sentiment_module.set_option("TIME_FILTER", self.get_option("TIME_FILTER"))
        sentiment_module.set_option("POST_LIMIT", self.get_option("POST_LIMIT"))
        sentiment_module.set_option("MIN_MENTIONS", self.get_option("MIN_MENTIONS"))

        # Get specific symbol if set
        symbol = self.get_option("SYMBOL")
        if symbol:
            sentiment_module.set_option("FILTER_SYMBOL", symbol)

        print("Fetching Reddit sentiment data...")
        sentiment_results = sentiment_module.run()

        if not sentiment_results.get('success'):
            return sentiment_results

        # Get strategy parameters
        entry_threshold = float(self.get_option("SENTIMENT_ENTRY_THRESHOLD"))
        exit_threshold = float(self.get_option("SENTIMENT_EXIT_THRESHOLD"))
        min_post_score = float(self.get_option("MIN_POST_SCORE"))
        max_negative_ratio = float(self.get_option("MAX_NEGATIVE_RATIO"))
        use_price_confirmation = self.get_option("USE_PRICE_CONFIRMATION")

        # Analyze sentiment and generate signals
        signals = []
        top_picks = []

        for ticker_data in sentiment_results['results']:
            ticker = ticker_data['ticker']
            sentiment_score = ticker_data['avg_sentiment']
            mentions = ticker_data['mentions']
            avg_post_score = ticker_data['avg_post_score']
            negative_ratio = ticker_data['negative_mentions'] / mentions if mentions > 0 else 0

            # Generate signal
            signal = self._generate_signal(
                ticker, sentiment_score, mentions, avg_post_score,
                negative_ratio, entry_threshold, exit_threshold,
                min_post_score, max_negative_ratio
            )

            if signal:
                signals.append(signal)

                # Add price data if confirmation needed
                if use_price_confirmation and signal['action'] == 'BUY':
                    price_signal = self._check_price_confirmation(ticker)
                    signal['price_confirmation'] = price_signal

                    # Only add to top picks if price confirms
                    if price_signal['confirmed']:
                        signal['combined_score'] = (
                            signal['signal_strength'] * 0.6 +
                            price_signal['momentum_score'] * 0.4
                        )
                        top_picks.append(signal)
                elif signal['action'] == 'BUY':
                    signal['combined_score'] = signal['signal_strength']
                    top_picks.append(signal)

        # Sort top picks by combined score
        top_picks.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

        # If specific symbol requested, provide detailed analysis
        if symbol:
            return self._analyze_specific_symbol(
                symbol, sentiment_results, signals,
                entry_threshold, exit_threshold
            )

        # Return overall analysis
        return {
            'success': True,
            'timestamp': sentiment_results['timestamp'],
            'strategy': 'Reddit Sentiment Trading',
            'subreddits_analyzed': sentiment_results['subreddits_analyzed'],
            'total_tickers': len(signals),
            'buy_signals': len([s for s in signals if s['action'] == 'BUY']),
            'sell_signals': len([s for s in signals if s['action'] == 'SELL']),
            'hold_signals': len([s for s in signals if s['action'] == 'HOLD']),
            'top_picks': top_picks[:10],
            'all_signals': signals,
            'sentiment_summary': sentiment_results['top_10_tickers']
        }

    def _generate_signal(self, ticker: str, sentiment: float, mentions: int,
                        avg_post_score: float, negative_ratio: float,
                        entry_threshold: float, exit_threshold: float,
                        min_post_score: float, max_negative_ratio: float) -> Dict[str, Any]:
        """Generate trading signal based on sentiment metrics"""

        # Check entry conditions
        buy_conditions = [
            sentiment >= entry_threshold,
            avg_post_score >= min_post_score,
            negative_ratio <= max_negative_ratio
        ]

        # Calculate signal strength (0-100)
        strength = 0
        reasons = []

        if sentiment >= 0.5:
            strength += 40
            reasons.append("Very bullish sentiment")
        elif sentiment >= 0.3:
            strength += 25
            reasons.append("Bullish sentiment")
        elif sentiment >= 0.1:
            strength += 10
            reasons.append("Slightly bullish sentiment")

        if mentions >= 50:
            strength += 30
            reasons.append(f"High mention volume ({mentions})")
        elif mentions >= 20:
            strength += 20
            reasons.append(f"Moderate mention volume ({mentions})")
        elif mentions >= 10:
            strength += 10
            reasons.append(f"Decent mention volume ({mentions})")

        if avg_post_score >= 100:
            strength += 20
            reasons.append("High quality discussions")
        elif avg_post_score >= 50:
            strength += 10
            reasons.append("Good quality discussions")

        if negative_ratio < 0.1:
            strength += 10
            reasons.append("Very low negative sentiment")

        # Determine action
        if all(buy_conditions) and strength >= 50:
            action = "BUY"
            confidence = "HIGH" if strength >= 70 else "MEDIUM"
        elif sentiment > exit_threshold and strength >= 30:
            action = "HOLD"
            confidence = "MEDIUM" if strength >= 50 else "LOW"
        else:
            action = "SELL"
            confidence = "HIGH" if sentiment < 0 else "MEDIUM"
            if sentiment < 0:
                reasons = ["Negative sentiment detected"]
            elif sentiment < exit_threshold:
                reasons = ["Sentiment below exit threshold"]

        return {
            'ticker': ticker,
            'action': action,
            'confidence': confidence,
            'signal_strength': strength,
            'sentiment_score': sentiment,
            'mentions': mentions,
            'avg_post_score': avg_post_score,
            'negative_ratio': negative_ratio,
            'reasons': reasons
        }

    def _check_price_confirmation(self, ticker: str) -> Dict[str, Any]:
        """Check if price action confirms sentiment signal"""
        try:
            fetcher = DataFetcher(self.framework.database)
            df = fetcher.get_stock_data(ticker, period="1mo", interval="1d")

            if df is None or df.empty or len(df) < 20:
                return {'confirmed': False, 'momentum_score': 0, 'reason': 'Insufficient data'}

            # Calculate short-term momentum
            roc_5 = ((df['Close'].iloc[-1] / df['Close'].iloc[-6]) - 1) * 100
            roc_10 = ((df['Close'].iloc[-1] / df['Close'].iloc[-11]) - 1) * 100

            # Volume trend
            avg_volume_recent = df['Volume'].iloc[-5:].mean()
            avg_volume_older = df['Volume'].iloc[-20:-5].mean()
            volume_increase = (avg_volume_recent / avg_volume_older - 1) * 100

            # Momentum score
            momentum_score = 0
            confirmed = False
            reasons = []

            if roc_5 > 3:
                momentum_score += 40
                reasons.append(f"Strong 5-day momentum (+{roc_5:.1f}%)")
                confirmed = True
            elif roc_5 > 0:
                momentum_score += 20
                reasons.append(f"Positive 5-day momentum (+{roc_5:.1f}%)")

            if roc_10 > 5:
                momentum_score += 30
                reasons.append(f"Strong 10-day momentum (+{roc_10:.1f}%)")

            if volume_increase > 50:
                momentum_score += 30
                reasons.append(f"Volume surge (+{volume_increase:.0f}%)")
                confirmed = True
            elif volume_increase > 20:
                momentum_score += 15
                reasons.append(f"Volume increase (+{volume_increase:.0f}%)")

            if not reasons:
                reasons.append("No price momentum confirmation")

            return {
                'confirmed': confirmed or momentum_score >= 50,
                'momentum_score': min(momentum_score, 100),
                'roc_5d': round(roc_5, 2),
                'roc_10d': round(roc_10, 2),
                'volume_change': round(volume_increase, 2),
                'reasons': reasons
            }

        except Exception as e:
            return {'confirmed': False, 'momentum_score': 0, 'reason': f'Error: {str(e)}'}

    def _analyze_specific_symbol(self, symbol: str, sentiment_results: Dict[str, Any],
                                 signals: List[Dict[str, Any]], entry_threshold: float,
                                 exit_threshold: float) -> Dict[str, Any]:
        """Provide detailed analysis for specific symbol"""

        # Find symbol in results
        ticker_data = None
        for result in sentiment_results['results']:
            if result['ticker'] == symbol:
                ticker_data = result
                break

        if not ticker_data:
            return {
                'success': False,
                'error': f'No sentiment data found for {symbol}. Symbol may not be mentioned on Reddit.'
            }

        # Find signal
        signal = None
        for s in signals:
            if s['ticker'] == symbol:
                signal = s
                break

        # Get price data
        fetcher = DataFetcher(self.framework.database)
        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        df = fetcher.get_stock_data(symbol, period, interval)

        # Build detailed response
        return {
            'success': True,
            'symbol': symbol,
            'current_price': df['Close'].iloc[-1] if df is not None and not df.empty else None,
            'sentiment_data': ticker_data,
            'trading_signal': signal,
            'price_data': df.tail(30) if df is not None else None,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'recommendation': self._generate_recommendation(ticker_data, signal, entry_threshold)
        }

    def _generate_recommendation(self, ticker_data: Dict[str, Any],
                                signal: Dict[str, Any],
                                entry_threshold: float) -> str:
        """Generate human-readable recommendation"""

        if not signal:
            return "WAIT - Insufficient data for signal generation"

        action = signal['action']
        sentiment = ticker_data['avg_sentiment']
        mentions = ticker_data['mentions']

        if action == 'BUY':
            return (f"🟢 BUY SIGNAL: {ticker_data['ticker']} has {ticker_data['sentiment_category']} "
                   f"sentiment (score: {sentiment:.3f}) with {mentions} mentions. "
                   f"Signal strength: {signal['signal_strength']}/100. "
                   f"Reasons: {', '.join(signal['reasons'])}")
        elif action == 'HOLD':
            return (f"🟡 HOLD: {ticker_data['ticker']} has moderate sentiment but doesn't meet "
                   f"strong entry criteria. Monitor for increased positive sentiment.")
        else:
            return (f"🔴 SELL/AVOID: {ticker_data['ticker']} has weak or negative sentiment. "
                   f"Not recommended for entry.")
