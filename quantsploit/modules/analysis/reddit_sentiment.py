"""
Reddit Sentiment Analysis Module
Spider Reddit for stock tickers and analyze sentiment
"""

import pandas as pd
import re
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from quantsploit.core.module import BaseModule
from collections import defaultdict, Counter

try:
    import praw
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False


class RedditSentiment(BaseModule):
    """
    Spider Reddit for stock tickers and analyze sentiment based on text content
    """

    # Common stock ticker pattern
    TICKER_PATTERN = re.compile(r'\b[A-Z]{1,5}\b')

    # Words/phrases that indicate we're NOT talking about a stock
    EXCLUDE_WORDS = {
        'THE', 'A', 'I', 'AM', 'ARE', 'IS', 'WAS', 'WERE', 'BE', 'BEEN',
        'DD', 'YOLO', 'WSB', 'CEO', 'CFO', 'IPO', 'PE', 'EPS', 'ATH', 'ATL',
        'EDIT', 'TLDR', 'TL', 'DR', 'IMO', 'IMHO', 'OP', 'FOMO', 'FUD',
        'USA', 'US', 'UK', 'EU', 'CEO', 'CTO', 'VP', 'ETF', 'SPY', 'QQQ',
        'IT', 'AI', 'ML', 'API', 'CEO', 'CPA', 'IRS', 'SEC', 'GDP', 'CPI',
        'PM', 'AM', 'UTC', 'EST', 'PST', 'MST', 'CST', 'NOT', 'AND', 'OR',
        'IF', 'BUT', 'SO', 'AS', 'AT', 'BY', 'FOR', 'FROM', 'IN', 'OF', 'ON',
        'TO', 'UP', 'OUT', 'MY', 'YOUR', 'ALL', 'NEW', 'OLD', 'NOW', 'JUST'
    }

    @property
    def name(self) -> str:
        return "Reddit Sentiment Analyzer"

    @property
    def description(self) -> str:
        return "Spider Reddit for stock ticker mentions and analyze sentiment"

    @property
    def author(self) -> str:
        return "Quantsploit Team"

    @property
    def category(self) -> str:
        return "analysis"

    def _init_options(self):
        super()._init_options()
        self.options.update({
            "SUBREDDITS": {
                "value": "wallstreetbets,stocks,investing,StockMarket,options",
                "required": False,
                "description": "Comma-separated list of subreddits to analyze"
            },
            "TIME_FILTER": {
                "value": "day",
                "required": False,
                "description": "Time filter: hour, day, week, month, year, all"
            },
            "POST_LIMIT": {
                "value": 100,
                "required": False,
                "description": "Number of posts to fetch per subreddit"
            },
            "MIN_SCORE": {
                "value": 10,
                "required": False,
                "description": "Minimum upvote score for posts"
            },
            "ANALYZE_COMMENTS": {
                "value": True,
                "required": False,
                "description": "Also analyze comments (more comprehensive but slower)"
            },
            "COMMENT_LIMIT": {
                "value": 50,
                "required": False,
                "description": "Max comments to analyze per post"
            },
            "FILTER_SYMBOL": {
                "value": "",
                "required": False,
                "description": "Only analyze specific symbol (leave empty for all tickers)"
            },
            "MIN_MENTIONS": {
                "value": 3,
                "required": False,
                "description": "Minimum mentions required to include ticker in results"
            }
        })

    def run(self) -> Dict[str, Any]:
        """Execute Reddit sentiment analysis"""
        if not REDDIT_AVAILABLE:
            return {
                "success": False,
                "error": "Reddit dependencies not installed. Run: pip install praw vaderSentiment"
            }

        # Get options
        subreddits_str = self.get_option("SUBREDDITS")
        time_filter = self.get_option("TIME_FILTER")
        post_limit = int(self.get_option("POST_LIMIT"))
        min_score = int(self.get_option("MIN_SCORE"))
        analyze_comments = self.get_option("ANALYZE_COMMENTS")
        comment_limit = int(self.get_option("COMMENT_LIMIT"))
        filter_symbol = self.get_option("FILTER_SYMBOL").strip().upper()
        min_mentions = int(self.get_option("MIN_MENTIONS"))

        subreddits = [s.strip() for s in subreddits_str.split(",")]

        # Initialize Reddit API (read-only mode - no credentials needed)
        try:
            reddit = praw.Reddit(
                client_id="reddit_sentiment_bot",
                client_secret=None,
                user_agent="Quantsploit Sentiment Analyzer v1.0"
            )
            # Test connection by accessing a public endpoint
            reddit.read_only = True
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to initialize Reddit API: {str(e)}\n"
                        "Note: PRAW now requires Reddit API credentials.\n"
                        "Get credentials at: https://www.reddit.com/prefs/apps\n"
                        "Set environment variables: REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET"
            }

        # Initialize sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()

        # Collect all mentions and their sentiments
        ticker_data = defaultdict(lambda: {
            'mentions': 0,
            'sentiment_scores': [],
            'posts': [],
            'average_score': 0,
            'positive_mentions': 0,
            'negative_mentions': 0,
            'neutral_mentions': 0
        })

        total_posts_analyzed = 0
        total_comments_analyzed = 0

        self.print_info(f"Analyzing {len(subreddits)} subreddit(s)...")

        # Spider Reddit
        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                self.print_info(f"Fetching posts from r/{subreddit_name}...")

                # Get top posts based on time filter
                posts = subreddit.top(time_filter=time_filter, limit=post_limit)

                for post in posts:
                    # Skip low-scoring posts
                    if post.score < min_score:
                        continue

                    total_posts_analyzed += 1

                    # Extract tickers and analyze sentiment from post title and body
                    post_text = f"{post.title} {post.selftext}"
                    tickers = self._extract_tickers(post_text)

                    # Get sentiment for the entire post
                    sentiment = analyzer.polarity_scores(post_text)

                    for ticker in tickers:
                        # Apply filter if specified
                        if filter_symbol and ticker != filter_symbol:
                            continue

                        ticker_data[ticker]['mentions'] += 1
                        ticker_data[ticker]['sentiment_scores'].append(sentiment['compound'])
                        ticker_data[ticker]['average_score'] += post.score

                        # Categorize sentiment
                        if sentiment['compound'] >= 0.05:
                            ticker_data[ticker]['positive_mentions'] += 1
                        elif sentiment['compound'] <= -0.05:
                            ticker_data[ticker]['negative_mentions'] += 1
                        else:
                            ticker_data[ticker]['neutral_mentions'] += 1

                        # Store post info
                        ticker_data[ticker]['posts'].append({
                            'title': post.title[:100],
                            'score': post.score,
                            'url': f"https://reddit.com{post.permalink}",
                            'created': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M'),
                            'sentiment': sentiment['compound'],
                            'subreddit': subreddit_name
                        })

                    # Analyze comments if enabled
                    if analyze_comments:
                        try:
                            post.comments.replace_more(limit=0)  # Remove "MoreComments" objects
                            comments = list(post.comments)[:comment_limit]

                            for comment in comments:
                                total_comments_analyzed += 1
                                comment_tickers = self._extract_tickers(comment.body)
                                comment_sentiment = analyzer.polarity_scores(comment.body)

                                for ticker in comment_tickers:
                                    if filter_symbol and ticker != filter_symbol:
                                        continue

                                    ticker_data[ticker]['mentions'] += 1
                                    ticker_data[ticker]['sentiment_scores'].append(comment_sentiment['compound'])

                                    if comment_sentiment['compound'] >= 0.05:
                                        ticker_data[ticker]['positive_mentions'] += 1
                                    elif comment_sentiment['compound'] <= -0.05:
                                        ticker_data[ticker]['negative_mentions'] += 1
                                    else:
                                        ticker_data[ticker]['neutral_mentions'] += 1

                        except Exception as e:
                            # Skip comment analysis for this post if error
                            pass

            except Exception as e:
                self.print_error(f"Error analyzing r/{subreddit_name}: {str(e)}")
                continue

        # Calculate aggregate metrics
        results = []
        for ticker, data in ticker_data.items():
            if data['mentions'] < min_mentions:
                continue

            avg_sentiment = sum(data['sentiment_scores']) / len(data['sentiment_scores'])
            avg_post_score = data['average_score'] / data['mentions']

            # Sort posts by score
            top_posts = sorted(data['posts'], key=lambda x: x['score'], reverse=True)[:5]

            results.append({
                'ticker': ticker,
                'mentions': data['mentions'],
                'avg_sentiment': round(avg_sentiment, 4),
                'sentiment_category': self._categorize_sentiment(avg_sentiment),
                'positive_mentions': data['positive_mentions'],
                'negative_mentions': data['negative_mentions'],
                'neutral_mentions': data['neutral_mentions'],
                'avg_post_score': round(avg_post_score, 2),
                'top_posts': top_posts,
                'sentiment_strength': self._calculate_strength(avg_sentiment, data['mentions'])
            })

        # Sort by mentions and sentiment strength
        results.sort(key=lambda x: (x['mentions'], abs(x['avg_sentiment'])), reverse=True)

        return {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'subreddits_analyzed': subreddits,
            'time_filter': time_filter,
            'total_posts': total_posts_analyzed,
            'total_comments': total_comments_analyzed,
            'tickers_found': len(results),
            'results': results,
            'top_10_tickers': results[:10]
        }

    def _extract_tickers(self, text: str) -> List[str]:
        """Extract potential stock tickers from text"""
        # Find all uppercase words
        potential_tickers = self.TICKER_PATTERN.findall(text)

        # Filter out common words and invalid tickers
        tickers = []
        for ticker in potential_tickers:
            if ticker not in self.EXCLUDE_WORDS and len(ticker) <= 5:
                tickers.append(ticker)

        return list(set(tickers))  # Remove duplicates

    def _categorize_sentiment(self, compound_score: float) -> str:
        """Categorize sentiment score"""
        if compound_score >= 0.5:
            return "Very Bullish"
        elif compound_score >= 0.25:
            return "Bullish"
        elif compound_score >= 0.05:
            return "Slightly Bullish"
        elif compound_score <= -0.5:
            return "Very Bearish"
        elif compound_score <= -0.25:
            return "Bearish"
        elif compound_score <= -0.05:
            return "Slightly Bearish"
        else:
            return "Neutral"

    def _calculate_strength(self, sentiment: float, mentions: int) -> float:
        """Calculate sentiment strength based on score and volume"""
        # Strength = |sentiment| * log(mentions)
        # This gives more weight to tickers with many mentions
        import math
        return round(abs(sentiment) * math.log(max(mentions, 2)), 4)

    def show_results(self, results: Dict[str, Any]):
        """Display results in a formatted way"""
        if not results.get('success'):
            self.print_error(results.get('error', 'Unknown error'))
            return

        self.print_good(f"\n{'='*80}")
        self.print_good(f"Reddit Sentiment Analysis Results")
        self.print_good(f"{'='*80}")
        self.print_info(f"Subreddits: {', '.join(results['subreddits_analyzed'])}")
        self.print_info(f"Time Filter: {results['time_filter']}")
        self.print_info(f"Posts Analyzed: {results['total_posts']}")
        self.print_info(f"Comments Analyzed: {results['total_comments']}")
        self.print_info(f"Tickers Found: {results['tickers_found']}")
        self.print_good(f"{'='*80}\n")

        # Display top 10 tickers
        self.print_good("Top 10 Most Mentioned Tickers with Sentiment:\n")

        header = f"{'Ticker':<8} {'Mentions':<10} {'Sentiment':<12} {'Category':<18} {'Strength':<10}"
        self.print_info(header)
        self.print_info("-" * len(header))

        for ticker_data in results['top_10_tickers']:
            ticker = ticker_data['ticker']
            mentions = ticker_data['mentions']
            sentiment = ticker_data['avg_sentiment']
            category = ticker_data['sentiment_category']
            strength = ticker_data['sentiment_strength']

            # Color code based on sentiment
            sentiment_str = f"{sentiment:+.4f}"
            if sentiment >= 0.05:
                sentiment_color = "green"
            elif sentiment <= -0.05:
                sentiment_color = "red"
            else:
                sentiment_color = "yellow"

            row = f"{ticker:<8} {mentions:<10} {sentiment_str:<12} {category:<18} {strength:<10.4f}"

            if sentiment >= 0.05:
                self.print_good(row)
            elif sentiment <= -0.05:
                self.print_error(row)
            else:
                self.print_warning(row)

        self.print_good(f"\n{'='*80}\n")
