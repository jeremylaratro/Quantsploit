"""
Reddit Sentiment Analysis Module
Spider Reddit for stock tickers and analyze sentiment
"""

import os
import math
import pandas as pd
import re
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from quantsploit.core.module import BaseModule
from collections import defaultdict, Counter

try:
    import praw
    import prawcore
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

import requests


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

    POSITIVE_CUES = {
        "bullish", "moon", "mooning", "uptrend", "surge", "beat", "beats", "beating",
        "strong", "stronger", "strength", "gain", "gains", "green", "rally", "pumping",
        "undervalued", "cheap", "breakout", "rip", "ripping", "squeeze", "alpha",
        "run", "running", "momentum", "profit", "profits", "profitable"
    }

    NEGATIVE_CUES = {
        "bearish", "dump", "dumping", "crash", "crashing", "downtrend", "selloff",
        "weak", "weaker", "weakness", "loss", "losses", "red", "bagholder", "bags",
        "overvalued", "expensive", "collapse", "collapsing", "dilution", "bankrupt",
        "fraud", "scam", "dead", "bleeding", "short", "shorting"
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
            "SORT": {
                "value": "top",
                "required": False,
                "description": "Sort: top, new, hot"
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
            },
            "ACCESS_MODE": {
                "value": "auto",
                "required": False,
                "description": "auto (API if creds else scrape), api, or scrape (JSON from old.reddit.com)"
            }
        })

    def run(self) -> Dict[str, Any]:
        """Execute Reddit sentiment analysis"""
        if not VADER_AVAILABLE:
            return {
                "success": False,
                "error": "Sentiment dependency not installed. Run: pip install vaderSentiment"
            }

        # Get options
        subreddits_str = self.get_option("SUBREDDITS")
        time_filter = self.get_option("TIME_FILTER")
        post_limit = int(self.get_option("POST_LIMIT"))
        sort = str(self.get_option("SORT")).lower()
        min_score = int(self.get_option("MIN_SCORE"))
        analyze_comments = self.get_option("ANALYZE_COMMENTS")
        comment_limit = int(self.get_option("COMMENT_LIMIT"))
        filter_symbol = self.get_option("FILTER_SYMBOL").strip().upper()
        min_mentions = int(self.get_option("MIN_MENTIONS"))
        access_mode = str(self.get_option("ACCESS_MODE")).lower()

        if sort not in {"top", "new", "hot"}:
            return {"success": False, "error": f"Invalid SORT '{sort}'. Use top, new, or hot."}

        allowed_time_filters = {"hour", "day", "week", "month", "year", "all"}
        if sort == "top" and time_filter not in allowed_time_filters:
            return {"success": False, "error": f"Invalid TIME_FILTER '{time_filter}'. Use one of {', '.join(sorted(allowed_time_filters))}."}
        elif sort != "top":
            # Non-top sorts ignore time_filter but we normalize it for reporting
            time_filter = "day"

        subreddits = [s.strip() for s in subreddits_str.split(",")]

        use_api = False
        reddit = None

        if access_mode == "api":
            use_api = True
        elif access_mode == "scrape":
            use_api = False
        else:  # auto
            use_api = bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"))

        if use_api and not PRAW_AVAILABLE:
            return {
                "success": False,
                "error": "ACCESS_MODE=api requires praw. Install with: pip install praw"
            }

        if use_api:
            try:
                reddit = self._initialize_reddit_client()
            except Exception as e:
                # If auto mode and API init fails, fall back to scrape
                if access_mode == "auto":
                    print(f"[WARN] API init failed ({e}); falling back to scraping.")
                    use_api = False
                else:
                    return {"success": False, "error": str(e)}
        else:
            if analyze_comments:
                print("[INFO] Comment analysis disabled in scrape mode.")
                analyze_comments = False

        # Initialize sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()

        # Collect all mentions and their sentiments
        ticker_data = defaultdict(lambda: {
            'mentions': 0,
            'sentiment_scores': [],
            'weighted_sentiment_sum': 0.0,
            'weight_sum': 0.0,
            'posts': [],
            'average_score': 0,
            'post_count': 0,
            'positive_mentions': 0,
            'negative_mentions': 0,
            'neutral_mentions': 0
        })

        total_posts_analyzed = 0
        total_comments_analyzed = 0

        print(f"Analyzing {len(subreddits)} subreddit(s)...")

        # Spider Reddit
        for subreddit_name in subreddits:
            try:
                print(f"Fetching posts from r/{subreddit_name} ({sort})...")

                if use_api:
                    subreddit = reddit.subreddit(subreddit_name)
                    if sort == "top":
                        posts = subreddit.top(time_filter=time_filter, limit=post_limit)
                    elif sort == "new":
                        posts = subreddit.new(limit=post_limit)
                    else:
                        posts = subreddit.hot(limit=post_limit)

                    for post in posts:
                        if post.score < min_score:
                            continue

                        total_posts_analyzed += 1
                        post_text = f"{post.title} {post.selftext}"
                        tickers = self._extract_tickers(post_text)

                        for ticker in tickers:
                            if filter_symbol and ticker != filter_symbol:
                                continue

                            mention_score = self._score_ticker_sentiment(
                                ticker, post.title, post.selftext, analyzer
                            )
                            weight = self._calculate_quality_weight(post.score)
                            if ticker in post.title:
                                weight *= 1.2  # emphasize explicit title mentions

                            self._accumulate_ticker_data(
                                ticker_data,
                                ticker,
                                mention_score,
                                weight,
                                post_score=post.score,
                                post_info={
                                    'title': post.title[:100],
                                    'score': post.score,
                                    'url': f"https://reddit.com{post.permalink}",
                                    'created': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M'),
                                    'sentiment': mention_score,
                                    'subreddit': subreddit_name
                                }
                            )

                        if analyze_comments:
                            try:
                                post.comments.replace_more(limit=0)
                                comments = list(post.comments)[:comment_limit]

                                for comment in comments:
                                    total_comments_analyzed += 1
                                    comment_tickers = self._extract_tickers(comment.body)
                                    comment_score = getattr(comment, "score", 0)

                                    for ticker in comment_tickers:
                                        if filter_symbol and ticker != filter_symbol:
                                            continue

                                        mention_score = self._score_text_with_cues(comment.body, analyzer)
                                        weight = 0.6 * self._calculate_quality_weight(comment_score)

                                        self._accumulate_ticker_data(
                                            ticker_data,
                                            ticker,
                                            mention_score,
                                            weight
                                        )

                            except Exception:
                                pass
                else:
                    scraped_posts = self._scrape_subreddit_json(subreddit_name, sort, time_filter, post_limit)

                    for post in scraped_posts:
                        if post["score"] < min_score:
                            continue

                        total_posts_analyzed += 1
                        post_text = f"{post['title']} {post['selftext']}"
                        tickers = self._extract_tickers(post_text)

                        for ticker in tickers:
                            if filter_symbol and ticker != filter_symbol:
                                continue

                            mention_score = self._score_ticker_sentiment(
                                ticker, post["title"], post["selftext"], analyzer
                            )
                            weight = self._calculate_quality_weight(post["score"])
                            if ticker in post["title"]:
                                weight *= 1.2

                            self._accumulate_ticker_data(
                                ticker_data,
                                ticker,
                                mention_score,
                                weight,
                                post_score=post["score"],
                                post_info={
                                    'title': post["title"][:100],
                                    'score': post["score"],
                                    'url': post["url"],
                                    'created': datetime.fromtimestamp(post["created"]).strftime('%Y-%m-%d %H:%M'),
                                    'sentiment': mention_score,
                                    'subreddit': subreddit_name
                                }
                            )

            except Exception as e:
                print(f"Error analyzing r/{subreddit_name}: {str(e)}")
                continue

        # Calculate aggregate metrics
        results = []
        for ticker, data in ticker_data.items():
            if data['mentions'] < min_mentions:
                continue

            if data['weight_sum'] > 0:
                avg_sentiment = data['weighted_sentiment_sum'] / data['weight_sum']
            elif data['sentiment_scores']:
                avg_sentiment = sum(data['sentiment_scores']) / len(data['sentiment_scores'])
            else:
                avg_sentiment = 0
            avg_post_score = data['average_score'] / max(data['post_count'], 1)

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
            'sort': sort,
            'access_mode': "api" if use_api else "scrape",
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
        return round(abs(sentiment) * math.log(max(mentions, 2)), 4)

    def show_results(self, results: Dict[str, Any]):
        """Display results in a formatted way"""
        if not results.get('success'):
            print(f"Error: {results.get('error', 'Unknown error')}")
            return

        print(f"\n{'='*80}")
        print(f"Reddit Sentiment Analysis Results")
        print(f"{'='*80}")
        print(f"Subreddits: {', '.join(results['subreddits_analyzed'])}")
        print(f"Sort: {results.get('sort', 'top')} | Time Filter: {results['time_filter']}")
        print(f"Access Mode: {results.get('access_mode', 'api')}")
        print(f"Posts Analyzed: {results['total_posts']}")
        print(f"Comments Analyzed: {results['total_comments']}")
        print(f"Tickers Found: {results['tickers_found']}")
        print(f"{'='*80}\n")

        # Display top 10 tickers
        print("Top 10 Most Mentioned Tickers with Sentiment:\n")

        header = f"{'Ticker':<8} {'Mentions':<10} {'Sentiment':<12} {'Category':<18} {'Strength':<10}"
        print(header)
        print("-" * len(header))

        for ticker_data in results['top_10_tickers']:
            ticker = ticker_data['ticker']
            mentions = ticker_data['mentions']
            sentiment = ticker_data['avg_sentiment']
            category = ticker_data['sentiment_category']
            strength = ticker_data['sentiment_strength']

            sentiment_str = f"{sentiment:+.4f}"
            row = f"{ticker:<8} {mentions:<10} {sentiment_str:<12} {category:<18} {strength:<10.4f}"
            print(row)

        print(f"\n{'='*80}\n")

    def _initialize_reddit_client(self):
        """
        Configure and validate a read-only Reddit client.

        Reddit now requires real API credentials even for read-only usage.
        We fail fast with clear guidance so users can fix 401 errors quickly.
        """
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv(
            "REDDIT_USER_AGENT",
            "Quantsploit Reddit Sentiment (by u/your_username)"
        )

        missing = []
        if not client_id:
            missing.append("REDDIT_CLIENT_ID")
        if not client_secret:
            missing.append("REDDIT_CLIENT_SECRET")

        if missing:
            raise RuntimeError(
                "Reddit API credentials are required. "
                f"Missing environment variable(s): {', '.join(missing)}. "
                "Create a Reddit app (type: script) at https://www.reddit.com/prefs/apps "
                "and export REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and optionally REDDIT_USER_AGENT."
            )

        try:
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                check_for_async=False,
            )
            reddit.read_only = True

            # Trigger a lightweight request to surface auth issues (e.g., 401) immediately
            try:
                next(reddit.subreddit("all").hot(limit=1))
            except StopIteration:
                pass

            return reddit
        except prawcore.exceptions.ResponseException as exc:
            if exc.response.status_code == 401:
                raise RuntimeError(
                    "Received 401 Unauthorized from Reddit. "
                    "Verify REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET, ensure the app type is 'script', "
                    "and regenerate the secret if it was reset."
                )
            raise RuntimeError(f"Reddit API error: {exc}")
        except prawcore.exceptions.OAuthException as exc:
            raise RuntimeError(
                f"Reddit OAuth error: {exc}. Double-check client id/secret and user agent."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Reddit API: {exc}")

    def _scrape_subreddit_json(self, subreddit: str, sort: str, time_filter: str, limit: int) -> List[Dict[str, Any]]:
        """
        Fetch posts using the public JSON endpoints on old.reddit.com.
        This avoids the authenticated API but still provides post bodies.
        """
        base_url = f"https://old.reddit.com/r/{subreddit}/{sort}/.json"
        params = {"limit": limit}
        if sort == "top":
            params["t"] = time_filter

        headers = {
            "User-Agent": os.getenv(
                "REDDIT_USER_AGENT",
                "Quantsploit Reddit Sentiment (scrape mode)"
            )
        }

        resp = requests.get(base_url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            raise RuntimeError("Rate limited by Reddit. Slow down requests or try again later.")
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to scrape r/{subreddit} ({resp.status_code}).")

        data = resp.json()
        posts = []

        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            posts.append({
                "title": post.get("title", ""),
                "selftext": post.get("selftext", "") or "",
                "score": post.get("score", 0),
                "created": post.get("created_utc", datetime.utcnow().timestamp()),
                "url": f"https://www.reddit.com{post.get('permalink', '')}",
            })

        return posts

    def _split_sentences(self, text: str) -> List[str]:
        """Lightweight sentence splitter to avoid heavy dependencies."""
        parts = re.split(r'[.!?;\n]+', text)
        return [p.strip() for p in parts if p and len(p.strip()) > 2]

    def _score_text_with_cues(self, text: str, analyzer) -> float:
        """Base VADER score with finance-specific lexical adjustments."""
        base = analyzer.polarity_scores(text)['compound']
        tokens = re.findall(r"[A-Za-z']+", text.lower())

        boost = 0.0
        for token in tokens:
            if token in self.POSITIVE_CUES:
                boost += 0.02
            if token in self.NEGATIVE_CUES:
                boost -= 0.02

        exclamation_boost = min(text.count('!'), 3) * 0.02
        boost += exclamation_boost

        score = max(-1.0, min(1.0, base + max(-0.15, min(0.15, boost))))
        return score

    def _score_ticker_sentiment(self, ticker: str, title: str, body: str, analyzer) -> float:
        """
        Score sentiment for a specific ticker using its sentence-level context,
        with a fallback to whole-text scoring.
        """
        title = title or ""
        body = body or ""
        combined = f"{title} {body}".strip()

        sentences = self._split_sentences(combined)
        contexts = [s for s in sentences if ticker in s.upper()]

        if not contexts:
            if title:
                contexts = [title]
            elif body:
                contexts = [body]
            else:
                contexts = [combined]

        scores = [self._score_text_with_cues(ctx, analyzer) for ctx in contexts if ctx]
        if not scores:
            return self._score_text_with_cues(combined, analyzer)
        return sum(scores) / len(scores)

    def _calculate_quality_weight(self, score: float) -> float:
        """Weight sentiment contributions by content quality (upvotes/score)."""
        return 1.0 + min(math.log1p(max(score, 0)) / 4.0, 1.5)

    def _accumulate_ticker_data(
        self,
        ticker_data: Dict[str, Any],
        ticker: str,
        mention_score: float,
        weight: float,
        post_score: float = None,
        post_info: Dict[str, Any] = None
    ):
        data = ticker_data[ticker]
        data['mentions'] += 1
        data['sentiment_scores'].append(mention_score)
        data['weighted_sentiment_sum'] += mention_score * weight
        data['weight_sum'] += weight

        if mention_score >= 0.05:
            data['positive_mentions'] += 1
        elif mention_score <= -0.05:
            data['negative_mentions'] += 1
        else:
            data['neutral_mentions'] += 1

        if post_score is not None:
            data['average_score'] += post_score
            data['post_count'] += 1

        if post_info:
            data['posts'].append(post_info)
