"""
Reddit Sentiment Analysis Module
Spider Reddit for stock tickers and analyze sentiment
"""

import os
import math
import pandas as pd
import re
from typing import Dict, Any, List, Tuple, Pattern
from datetime import datetime, timedelta
from quantsploit.core.module import BaseModule
from quantsploit.utils.ticker_validator import get_validator
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
        'USA', 'US', 'UK', 'EU', 'CEO', 'CTO', 'VP', 'ETF',
        'IT', 'AI', 'ML', 'API', 'CPA', 'IRS', 'SEC', 'GDP', 'CPI',
        'PM', 'AM', 'UTC', 'EST', 'PST', 'MST', 'CST', 'NOT', 'AND', 'OR',
        'IF', 'BUT', 'SO', 'AS', 'AT', 'BY', 'FOR', 'FROM', 'IN', 'OF', 'ON',
        'TO', 'UP', 'OUT', 'MY', 'YOUR', 'ALL', 'NEW', 'OLD', 'NOW', 'JUST',
        'AN', 'HE', 'SHE', 'WE', 'THEY', 'HAS', 'HAD', 'DO', 'DOES', 'DID',
        'WILL', 'WOULD', 'COULD', 'SHOULD', 'MAY', 'MIGHT', 'CAN', 'CANT',
        'EOD', 'AH', 'PM', 'ETA', 'FAQ', 'FYI', 'ASAP', 'RIP', 'LOL', 'LMAO',
        'OMG', 'WTF', 'SMH', 'TBH', 'IDK', 'IRL', 'NSFW', 'OC', 'OP', 'PSA'
    }

    # MASSIVELY EXPANDED POSITIVE SENTIMENT LEXICON
    # Covers: trading strategies, options, meme stock culture, momentum indicators
    POSITIVE_CUES = {
        # Core bullish terms
        "bullish", "bull", "bulls", "mega-bull", "super-bull", "uber-bull",

        # Moon/rocket terminology
        "moon", "mooning", "moonshot", "moonbound", "moons", "mooned",
        "rocket", "rockets", "rocketing", "rocketed", "rocketship",
        "🚀", "💎", "🌙", "📈", "🔥", "💰", "🤑",

        # Trend & momentum
        "uptrend", "uptrending", "upturn", "upside", "upmove", "upswing",
        "surge", "surging", "surged", "surges", "momentum", "breakout",
        "breaking-out", "broke-out", "breakthrough", "blast-off", "blastoff",

        # Performance & gains
        "beat", "beats", "beating", "crushed", "crushing", "smashed", "killed",
        "gain", "gains", "gaining", "gained", "gainer", "gainz", "tendies",
        "profit", "profits", "profitable", "profiting", "printing",
        "green", "greens", "greenday", "biggreen", "massive-gains",

        # Quality & value
        "strong", "stronger", "strongest", "strength", "solid", "robust",
        "healthy", "quality", "undervalued", "cheap", "discount", "bargain",
        "value", "steal", "opportunity", "gem", "hidden-gem", "sleeper",

        # Price action
        "rally", "rallying", "rallied", "rallies", "rip", "ripping", "ripped",
        "pump", "pumping", "pumped", "pumps", "squeeze", "squeezing", "squeezed",
        "run", "running", "runner", "runup", "run-up", "explode", "exploding",
        "pop", "popping", "popped", "spike", "spiking", "spiked", "skyrocket",

        # Fundamental positives
        "earnings-beat", "revenue-growth", "expansion", "innovation", "growth",
        "scalable", "profitable", "cash-flow", "positive-outlook", "upgrade",
        "outperform", "overperform", "beat-estimates", "guidance-raise",

        # Trading strategies (bullish)
        "long", "calls", "call", "buying-calls", "bought-calls", "itm", "otm",
        "leaps", "deep-itm", "going-long", "entered-long", "accumulating",
        "accumulation", "buying-dip", "btfd", "buy-the-dip", "dip-buying",
        "hold", "holding", "hodl", "hodling", "diamond-hands", "💎🙌",

        # Market psychology (positive)
        "confident", "conviction", "bullish-case", "thesis", "DD", "solid-dd",
        "backing-truck", "loading", "loaded", "loading-up", "doubling-down",
        "adding", "adding-more", "averaging-down", "buying-more",

        # Momentum indicators
        "volume-spike", "high-volume", "breaking-resistance", "new-high",
        "all-time-high", "ath", "52w-high", "gamma-squeeze", "short-squeeze",
        "shorts-squeezed", "covering", "forced-covering", "gap-up", "gapping",

        # Options/derivatives (bullish)
        "high-oi", "call-volume", "unusual-options", "smart-money", "whale-buy",
        "dark-pool-buying", "insider-buying", "institutional-buying",

        # Social sentiment
        "hype", "hyped", "trending", "viral", "fomo", "everyone-buying",
        "to-the-moon", "lambo", "lambos", "yacht", "retire", "retirement",

        # Technical analysis (bullish)
        "golden-cross", "ma-cross", "bullish-flag", "cup-and-handle",
        "ascending-triangle", "bull-flag", "higher-highs", "higher-lows",
        "macd-cross", "rsi-oversold", "bounce", "bouncing", "reversal",

        # Company-specific positives
        "partnership", "acquisition", "merger", "buyback", "dividend",
        "dividend-increase", "split", "stock-split", "spinoff", "ipo-success",

        # Misc positive
        "alpha", "outperforming", "leader", "dominance", "catalyst",
        "upcoming-catalyst", "news", "good-news", "positive", "optimistic",
        "hopium", "copium", "stonks", "stonk", "money-printer", "brrr"
    }

    # MASSIVELY EXPANDED NEGATIVE SENTIMENT LEXICON
    NEGATIVE_CUES = {
        # Core bearish terms
        "bearish", "bear", "bears", "mega-bear", "super-bearish",

        # Crash/collapse terminology
        "dump", "dumping", "dumped", "dumps", "crash", "crashing", "crashed",
        "collapse", "collapsing", "collapsed", "tank", "tanking", "tanked",
        "plunge", "plunging", "plunged", "crater", "cratering", "cratered",
        "📉", "💀", "🔴", "⚠️", "🩸",

        # Trend & momentum (negative)
        "downtrend", "downtrending", "downturn", "downside", "downmove",
        "selloff", "sell-off", "selling", "mass-selling", "capitulation",
        "bleeding", "bloodbath", "blood-red", "massacre", "slaughter",

        # Performance & losses
        "loss", "losses", "losing", "lost", "loser", "red", "reds", "bagholding",
        "bagholder", "bags", "heavy-bags", "caught-bag", "holding-bags",
        "miss", "missed", "missing", "disappointed", "disappointment",

        # Quality & value (negative)
        "weak", "weaker", "weakest", "weakness", "fragile", "vulnerable",
        "overvalued", "overpriced", "expensive", "bubble", "frothy", "stretched",
        "worthless", "trash", "garbage", "scam", "fraud", "ponzi", "rug",

        # Price action (negative)
        "drop", "dropping", "dropped", "fall", "falling", "fell", "decline",
        "declining", "declined", "sink", "sinking", "sunk", "tumble", "tumbling",
        "free-fall", "freefall", "nosedive", "cliff", "drill", "drilling",

        # Fundamental negatives
        "earnings-miss", "revenue-decline", "layoffs", "bankruptcy", "bankrupt",
        "chapter-11", "liquidation", "insolvent", "debt", "overleveraged",
        "cash-burn", "burning-cash", "dilution", "dilutive", "share-dilution",

        # Trading strategies (bearish)
        "short", "shorting", "shorted", "shorts", "puts", "put", "buying-puts",
        "bought-puts", "selling-calls", "sold-calls", "going-short",
        "entered-short", "fade", "fading", "faded", "selling", "sold", "exit",

        # Market psychology (negative)
        "fear", "panic", "panic-selling", "scared", "terrified", "worried",
        "concerned", "doubt", "doubting", "uncertain", "uncertainty",
        "hopeless", "despair", "giving-up", "capitulating", "throwing-towel",

        # Momentum indicators (negative)
        "low-volume", "no-volume", "breaking-support", "new-low", "52w-low",
        "all-time-low", "death-cross", "ma-death", "gap-down", "gapping-down",

        # Options/derivatives (bearish)
        "high-puts", "put-volume", "unusual-puts", "insider-selling",
        "institutional-selling", "dumping-shares", "offering", "secondary",

        # Social sentiment (negative)
        "fud", "spreading-fud", "doubt", "hate", "hated", "despised",
        "avoid", "stay-away", "warning", "red-flag", "red-flags", "concern",

        # Technical analysis (bearish)
        "death-cross", "bearish-flag", "head-and-shoulders", "double-top",
        "triple-top", "descending-triangle", "bear-flag", "lower-highs",
        "lower-lows", "breakdown", "breaking-down", "broke-down",

        # Company-specific negatives
        "investigation", "sec-investigation", "lawsuit", "sued", "fraud",
        "scandal", "controversy", "recall", "fine", "penalty", "regulation",
        "ban", "banned", "restricted", "delisted", "delisting",

        # Risk & warnings
        "risky", "risk", "dangerous", "caution", "warning", "trap", "bull-trap",
        "dead-cat", "dead-cat-bounce", "falling-knife", "catching-knife",

        # Misc negative
        "rekt", "wrecked", "destroyed", "demolished", "annihilated", "toast",
        "done", "finished", "over", "dead", "dying", "doomed", "failure",
        "fail", "failed", "rug-pull", "pump-and-dump", "exit-scam"
    }

    # REGEX PATTERNS FOR SPELLING VARIATIONS & SLANG
    # These patterns catch common misspellings, leetspeak, and deliberate variations
    POSITIVE_PATTERNS = [
        # Moon variations
        (re.compile(r'm[o0]{2,}n', re.I), 0.03),
        (re.compile(r'mo+ning', re.I), 0.03),
        (re.compile(r'ro+cket', re.I), 0.03),
        (re.compile(r'🚀+'), 0.02),

        # Diamond hands variations
        (re.compile(r'diamond.?hands?', re.I), 0.04),
        (re.compile(r'💎+.*🙌+'), 0.04),
        (re.compile(r'h[o0]dl', re.I), 0.03),

        # Bullish variations
        (re.compile(r'bu+l+ish', re.I), 0.03),
        (re.compile(r'bul+s?', re.I), 0.02),

        # Tendies/gains
        (re.compile(r'tendies?', re.I), 0.03),
        (re.compile(r'gainz+', re.I), 0.02),
        (re.compile(r'pr[o0]fit', re.I), 0.02),

        # Pump/squeeze
        (re.compile(r'pu+mp', re.I), 0.02),
        (re.compile(r'sque+ze', re.I), 0.03),
        (re.compile(r'sho+rt.?sque+ze', re.I), 0.04),

        # BTFD
        (re.compile(r'btfd', re.I), 0.03),
        (re.compile(r'buy.?the.?f.{0,3}dip', re.I), 0.03),

        # To the moon
        (re.compile(r'to+.?the+.?mo+n', re.I), 0.04),
        (re.compile(r'lambo+', re.I), 0.02),

        # Brrr (money printer)
        (re.compile(r'br+r+', re.I), 0.02),
        (re.compile(r'money.?printer', re.I), 0.02),

        # Calls/long
        (re.compile(r'buying.?cal+s', re.I), 0.02),
        (re.compile(r'long.?cal+s', re.I), 0.02),
    ]

    NEGATIVE_PATTERNS = [
        # Bearish variations
        (re.compile(r'be+a+rish', re.I), -0.03),
        (re.compile(r'be+a+rs?', re.I), -0.02),

        # Dump/crash
        (re.compile(r'du+mp', re.I), -0.03),
        (re.compile(r'cra+sh', re.I), -0.03),
        (re.compile(r'📉+'), -0.02),

        # Bagholder
        (re.compile(r'bag.?hold', re.I), -0.03),
        (re.compile(r'heavy.?bags', re.I), -0.03),
        (re.compile(r'holding.?bags', re.I), -0.03),

        # Rekt/wrecked
        (re.compile(r're+kt', re.I), -0.03),
        (re.compile(r'wre+cke*d', re.I), -0.03),

        # Paper hands
        (re.compile(r'paper.?hands?', re.I), -0.03),
        (re.compile(r'📄+.*🙌+'), -0.03),

        # Dead/dying
        (re.compile(r'de+a+d', re.I), -0.02),
        (re.compile(r'dying', re.I), -0.02),
        (re.compile(r'💀+'), -0.02),

        # Rug pull
        (re.compile(r'ru+g.?pu+l+', re.I), -0.04),
        (re.compile(r'pump.?and.?dump', re.I), -0.04),

        # FUD
        (re.compile(r'fu+d+', re.I), -0.02),
        (re.compile(r'spreading.?fud', re.I), -0.03),

        # Puts/short
        (re.compile(r'buying.?puts', re.I), -0.02),
        (re.compile(r'going.?short', re.I), -0.02),

        # Drilling/tanking
        (re.compile(r'dri+l+ing', re.I), -0.03),
        (re.compile(r'tanking', re.I), -0.03),
    ]

    # Negation words that flip sentiment
    NEGATION_WORDS = {
        # Basic negations
        'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere',

        # Contractions
        'isnt', "isn't", 'arent', "aren't", 'wasnt', "wasn't", 'werent', "weren't",
        'dont', "don't", 'doesnt', "doesn't", 'didnt', "didn't", 'wont', "won't",
        'wouldnt', "wouldn't", 'shouldnt', "shouldn't", 'cant', "can't", 'cannot',
        'couldnt', "couldn't", 'hasnt', "hasn't", 'havent', "haven't", 'hadnt', "hadn't",
        'mustnt', "mustn't", 'mightnt', "mightn't", 'shant', "shan't",

        # Quantifiers & modifiers
        'hardly', 'barely', 'scarcely', 'seldom', 'rarely',

        # Prepositions with negation sense
        'without', 'lacking', 'absent',

        # Negative conjunctions
        'nor',

        # Dismissals
        'doubt', 'doubtful', 'unlikely', 'impossible'
    }

    # HIGH-CONFIDENCE PHRASE PATTERNS
    # These phrases have strong sentiment signals and override normal scoring
    # Format: (pattern, sentiment_score, is_regex)
    # Sentiment score range: -1.0 (very negative) to +1.0 (very positive)
    HIGH_CONFIDENCE_PHRASES = [
        # Very negative phrases (personal loss/failure)
        (re.compile(r"it'?s (all )?over( for me)?", re.I), -0.75),
        (re.compile(r"i'?m (so |totally |completely )?done", re.I), -0.70),
        (re.compile(r"i'?m (so |totally |completely )?cooked", re.I), -0.75),
        (re.compile(r"(i'?m |got )?re+kt", re.I), -0.80),
        (re.compile(r"lost everything", re.I), -0.85),
        (re.compile(r"blew up my account", re.I), -0.90),
        (re.compile(r"account (is )?wiped", re.I), -0.85),
        (re.compile(r"need .{1,20} to break even", re.I), -0.65),  # "need MSTR to hit $320 to break even"
        (re.compile(r"(down|lost) \d+%", re.I), -0.60),
        (re.compile(r"bleeding (out|money)", re.I), -0.70),
        (re.compile(r"gave up", re.I), -0.65),
        (re.compile(r"throwing in the towel", re.I), -0.70),
        (re.compile(r"can'?t take (it|this) anymore", re.I), -0.75),

        # Negative financial phrases
        (re.compile(r"ponzi(-| )?(scheme|ratio)", re.I), -0.65),
        (re.compile(r"(total |complete )?rug pull", re.I), -0.80),
        (re.compile(r"exit scam", re.I), -0.85),
        (re.compile(r"going (to )?zero", re.I), -0.75),
        (re.compile(r"worthless", re.I), -0.70),

        # Very positive phrases (company dominance/success)
        (re.compile(r"(market |total |complete )?dominance", re.I), 0.65),
        (re.compile(r"crushing (it|the competition)", re.I), 0.70),
        (re.compile(r"absolute[ly]* dominating", re.I), 0.70),
        (re.compile(r"(best|strongest) in (the )?sector", re.I), 0.60),
        (re.compile(r"(clear|obvious) winner", re.I), 0.65),
        (re.compile(r"no competition", re.I), 0.60),
        (re.compile(r"(industry |market )?leader", re.I), 0.55),

        # Major gains/wins
        (re.compile(r"(took|locked in|realized) \d+%.*gains?", re.I), 0.75),
        (re.compile(r"up \d{2,}%", re.I), 0.60),  # "up 130%"
        (re.compile(r"\d+x gains?", re.I), 0.70),  # "10x gains"
        (re.compile(r"printing money", re.I), 0.65),
        (re.compile(r"(keep|still) printing", re.I), 0.60),
        (re.compile(r"life[- ]changing gains?", re.I), 0.80),
        (re.compile(r"lambo (soon|time|when)", re.I), 0.60),

        # Momentum/trajectory
        (re.compile(r"can'?t stop won'?t stop", re.I), 0.65),
        (re.compile(r"no stopping (it|this)", re.I), 0.60),
        (re.compile(r"unstoppable", re.I), 0.60),
        (re.compile(r"going parabolic", re.I), 0.70),
        (re.compile(r"straight (to|to the) moon", re.I), 0.65),
    ]

    # HIGH-CONFIDENCE TERM WEIGHTS
    # These specific terms get higher weights than the default 0.025
    # This helps overcome VADER misinterpretations
    HIGH_WEIGHT_POSITIVE = {
        "dominance": 0.15,
        "dominating": 0.15,
        "crushing": 0.12,
        "leader": 0.10,
        "winner": 0.10,
        "outperform": 0.12,
        "beat": 0.10,
        "beats": 0.10,
        "smashed": 0.12,
        "moonshot": 0.15,
        "gamma-squeeze": 0.15,
        "short-squeeze": 0.15,
    }

    HIGH_WEIGHT_NEGATIVE = {
        "over": 0.10,  # as in "it's over"
        "done": 0.10,  # as in "I'm done"
        "cooked": 0.15,
        "rekt": 0.15,
        "wrecked": 0.15,
        "destroyed": 0.12,
        "annihilated": 0.12,
        "worthless": 0.15,
        "scam": 0.12,
        "ponzi": 0.15,
        "rug": 0.12,
        "fraud": 0.12,
    }

    # CONTEXT-AWARE OUTCOME INDICATORS
    # When these appear near a ticker, amplify the sentiment
    POSITIVE_OUTCOMES = {
        "gains", "gain", "profit", "profits", "up", "rally", "rallying", "surge",
        "surging", "moon", "mooning", "rocket", "rocketing", "beat", "beats",
        "crushing", "smashed", "won", "winning", "success", "printing", "green",
        "dominance", "dominating", "leader", "outperform", "outperforming"
    }

    NEGATIVE_OUTCOMES = {
        "loss", "losses", "down", "crash", "crashing", "dump", "dumping", "tank",
        "tanking", "drop", "dropping", "fall", "falling", "miss", "missed",
        "disaster", "failed", "failing", "red", "bleeding", "collapse", "collapsing"
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
            },
            "VALIDATE_TICKERS": {
                "value": True,
                "required": False,
                "description": "Validate tickers against comprehensive database of valid symbols (recommended)"
            },
            "ADVANCED_SENTIMENT": {
                "value": True,
                "required": False,
                "description": "Use advanced sentiment analysis with regex patterns and negation detection"
            },
            "DEBUG_MODE": {
                "value": False,
                "required": False,
                "description": "Enable verbose debug output showing sentiment calculation details"
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
        validate_tickers = bool(self.get_option("VALIDATE_TICKERS"))
        advanced_sentiment = bool(self.get_option("ADVANCED_SENTIMENT"))
        debug_mode = bool(self.get_option("DEBUG_MODE"))

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
                        tickers = self._extract_tickers(post_text, validate_tickers)

                        for ticker in tickers:
                            if filter_symbol and ticker != filter_symbol:
                                continue

                            sentiment_result = self._score_ticker_sentiment(
                                ticker, post.title, post.selftext, analyzer, advanced_sentiment, debug_mode
                            )

                            # Handle debug mode where result is (score, debug_info)
                            if debug_mode:
                                mention_score, debug_info = sentiment_result
                            else:
                                mention_score = sentiment_result
                                debug_info = None

                            weight = self._calculate_quality_weight(post.score)
                            if ticker in post.title:
                                weight *= 1.2  # emphasize explicit title mentions

                            post_info = {
                                'title': post.title[:100],
                                'score': post.score,
                                'url': f"https://reddit.com{post.permalink}",
                                'created': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M'),
                                'sentiment': mention_score,
                                'subreddit': subreddit_name
                            }
                            if debug_info:
                                post_info['debug'] = debug_info

                            self._accumulate_ticker_data(
                                ticker_data,
                                ticker,
                                mention_score,
                                weight,
                                post_score=post.score,
                                post_info=post_info
                            )

                        if analyze_comments:
                            try:
                                post.comments.replace_more(limit=0)
                                comments = list(post.comments)[:comment_limit]

                                for comment in comments:
                                    total_comments_analyzed += 1
                                    comment_tickers = self._extract_tickers(comment.body, validate_tickers)
                                    comment_score = getattr(comment, "score", 0)

                                    for ticker in comment_tickers:
                                        if filter_symbol and ticker != filter_symbol:
                                            continue

                                        sentiment_result = self._score_text_with_cues(comment.body, analyzer, advanced_sentiment, debug_mode)

                                        # Handle debug mode
                                        if debug_mode:
                                            mention_score, _ = sentiment_result  # We don't store comment debug info
                                        else:
                                            mention_score = sentiment_result

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
                        tickers = self._extract_tickers(post_text, validate_tickers)

                        for ticker in tickers:
                            if filter_symbol and ticker != filter_symbol:
                                continue

                            sentiment_result = self._score_ticker_sentiment(
                                ticker, post["title"], post["selftext"], analyzer, advanced_sentiment, debug_mode
                            )

                            # Handle debug mode where result is (score, debug_info)
                            if debug_mode:
                                mention_score, debug_info = sentiment_result
                            else:
                                mention_score = sentiment_result
                                debug_info = None

                            weight = self._calculate_quality_weight(post["score"])
                            if ticker in post["title"]:
                                weight *= 1.2

                            post_info = {
                                'title': post["title"][:100],
                                'score': post["score"],
                                'url': post["url"],
                                'created': datetime.fromtimestamp(post["created"]).strftime('%Y-%m-%d %H:%M'),
                                'sentiment': mention_score,
                                'subreddit': subreddit_name
                            }
                            if debug_info:
                                post_info['debug'] = debug_info

                            self._accumulate_ticker_data(
                                ticker_data,
                                ticker,
                                mention_score,
                                weight,
                                post_score=post["score"],
                                post_info=post_info
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

    def _extract_tickers(self, text: str, validate: bool = True) -> List[str]:
        """
        Extract potential stock tickers from text with optional validation

        Args:
            text: Text to extract tickers from
            validate: Whether to validate against ticker database (default: True)

        Returns:
            List of valid ticker symbols
        """
        # Find all uppercase words
        potential_tickers = self.TICKER_PATTERN.findall(text)

        # Filter out common words and handle single-letter ticker collisions
        tickers = []
        for ticker in potential_tickers:
            if ticker not in self.EXCLUDE_WORDS and 1 <= len(ticker) <= 5:
                # Special handling for single-letter tickers to prevent false positives
                if len(ticker) == 1:
                    # Check if it's likely an abbreviation (e.g., "U.S.", "U.K.", "U.N.")
                    # by seeing if it's followed by a period in the original text
                    pattern = re.compile(rf'\b{re.escape(ticker)}\.')
                    if pattern.search(text):
                        # Skip this ticker - likely an abbreviation
                        continue

                tickers.append(ticker)

        # Remove duplicates
        unique_tickers = list(set(tickers))

        # Validate against ticker database if enabled
        if validate:
            try:
                validator = get_validator()
                valid_tickers = validator.validate_batch(unique_tickers)

                # Report invalid tickers for debugging (but not too spammy)
                invalid = validator.get_invalid_tickers(unique_tickers)
                if invalid:
                    # Only show first 5 to avoid spam
                    invalid_sample = sorted(invalid)[:5]
                    more = f" (+{len(invalid) - 5} more)" if len(invalid) > 5 else ""
                    if len(invalid) <= 15:  # Only log if not too many
                        print(f"  [Filtered] {', '.join(invalid_sample)}{more}")

                return valid_tickers
            except Exception as e:
                print(f"[WARN] Ticker validation failed: {e}, proceeding without validation")
                return unique_tickers

        return unique_tickers

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
        Fetch posts using the public JSON endpoints on old.reddit.com with pagination support.
        This avoids the authenticated API but still provides post bodies.
        Supports fetching more than 100 posts through pagination.
        """
        base_url = f"https://old.reddit.com/r/{subreddit}/{sort}/.json"
        headers = {
            "User-Agent": os.getenv(
                "REDDIT_USER_AGENT",
                "Quantsploit Reddit Sentiment (scrape mode)"
            )
        }

        all_posts = []
        after = None
        requests_made = 0
        max_requests = 10  # Safety limit to prevent infinite loops

        # Reddit returns max 100 posts per request, so we need pagination
        while len(all_posts) < limit and requests_made < max_requests:
            # Calculate how many more posts we need
            remaining = limit - len(all_posts)
            request_limit = min(100, remaining)  # Reddit's max is 100 per request

            params = {"limit": request_limit}
            if sort == "top":
                params["t"] = time_filter
            if after:
                params["after"] = after

            try:
                import time
                if requests_made > 0:
                    time.sleep(2)  # Rate limiting: wait 2 seconds between requests

                resp = requests.get(base_url, headers=headers, params=params, timeout=15)
                if resp.status_code == 429:
                    print(f"[WARN] Rate limited by Reddit, waiting 10 seconds...")
                    time.sleep(10)
                    continue
                if resp.status_code != 200:
                    print(f"[WARN] Failed to scrape r/{subreddit} ({resp.status_code}), stopping pagination")
                    break

                data = resp.json()
                children = data.get("data", {}).get("children", [])

                if not children:
                    # No more posts available
                    break

                for child in children:
                    post = child.get("data", {})
                    all_posts.append({
                        "title": post.get("title", ""),
                        "selftext": post.get("selftext", "") or "",
                        "score": post.get("score", 0),
                        "created": post.get("created_utc", datetime.utcnow().timestamp()),
                        "url": f"https://www.reddit.com{post.get('permalink', '')}",
                    })

                # Get the "after" token for pagination
                after = data.get("data", {}).get("after")
                requests_made += 1

                if not after:
                    # No more pages available
                    break

                print(f"[INFO] Fetched {len(all_posts)}/{limit} posts from r/{subreddit}...")

            except Exception as e:
                print(f"[WARN] Error during pagination: {e}, returning {len(all_posts)} posts")
                break

        print(f"[INFO] Total posts fetched from r/{subreddit}: {len(all_posts)}")
        return all_posts

    def _split_sentences(self, text: str) -> List[str]:
        """Lightweight sentence splitter to avoid heavy dependencies."""
        parts = re.split(r'[.!?;\n]+', text)
        return [p.strip() for p in parts if p and len(p.strip()) > 2]

    def _score_text_with_cues(self, text: str, analyzer, advanced: bool = True, debug: bool = False):
        """
        Advanced sentiment scoring with finance-specific lexical adjustments,
        regex pattern matching, and negation detection.

        Args:
            text: Text to analyze
            analyzer: VADER sentiment analyzer instance
            advanced: Use advanced features (regex, negation detection)
            debug: If True, return tuple of (score, debug_info), else just score

        Returns:
            Sentiment score between -1.0 and 1.0, or (score, debug_info) if debug=True
        """
        # Base VADER score
        base = analyzer.polarity_scores(text)['compound']

        if debug:
            print(f"\n{'='*80}")
            print(f"DEBUG: Sentiment Analysis")
            print(f"{'='*80}")
            print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
            print(f"Base VADER score: {base:.4f}")

        if not advanced:
            # Use simpler legacy scoring
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

        # ADVANCED SENTIMENT ANALYSIS
        boost = 0.0
        text_lower = text.lower()

        debug_info = {
            'positive_words': [],
            'negative_words': [],
            'positive_patterns': [],
            'negative_patterns': [],
            'negations': [],
            'emphasis': {},
            'high_confidence': None
        }

        # STEP 0: CHECK HIGH-CONFIDENCE PHRASES FIRST
        # These phrases override normal sentiment analysis
        for pattern, score in self.HIGH_CONFIDENCE_PHRASES:
            if pattern.search(text):
                if debug:
                    match = pattern.search(text).group()
                    debug_info['high_confidence'] = f"'{match}' → {score:.2f}"
                    print(f"\n{'='*80}")
                    print(f"DEBUG: HIGH-CONFIDENCE PHRASE DETECTED")
                    print(f"{'='*80}")
                    print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
                    print(f"Matched phrase: '{match}'")
                    print(f"Override score: {score:.2f}")
                    print(f"{'='*80}\n")
                return score  # Return immediately, bypassing all other analysis

        # 1. LEXICAL MATCHING - Check for sentiment cue words
        tokens = re.findall(r"[A-Za-z']+", text_lower)

        # Build bigrams and trigrams for better context
        words = text_lower.split()
        bigrams = [f"{words[i]}-{words[i+1]}" for i in range(len(words)-1)] if len(words) > 1 else []
        trigrams = [f"{words[i]}-{words[i+1]}-{words[i+2]}" for i in range(len(words)-2)] if len(words) > 2 else []

        # Check all n-grams against sentiment cues
        all_tokens = tokens + bigrams + trigrams
        for token in all_tokens:
            if token in self.POSITIVE_CUES:
                # Use high weight if available, otherwise default 0.025
                weight = self.HIGH_WEIGHT_POSITIVE.get(token, 0.025)
                boost += weight
                if debug:
                    debug_info['positive_words'].append(f"{token} (+{weight:.3f})")
            if token in self.NEGATIVE_CUES:
                # Use high weight if available, otherwise default 0.025
                weight = self.HIGH_WEIGHT_NEGATIVE.get(token, 0.025)
                boost -= weight
                if debug:
                    debug_info['negative_words'].append(f"{token} (-{weight:.3f})")

        # 2. REGEX PATTERN MATCHING - Catch spelling variations and slang
        for pattern, weight in self.POSITIVE_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                boost += weight * min(len(matches), 3)  # Cap at 3 matches per pattern
                if debug:
                    debug_info['positive_patterns'].append(f"{matches} ({weight * min(len(matches), 3):+.3f})")

        for pattern, weight in self.NEGATIVE_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                boost += weight * min(len(matches), 3)  # weight is already negative
                if debug:
                    debug_info['negative_patterns'].append(f"{matches} ({weight * min(len(matches), 3):+.3f})")

        # 3. NEGATION DETECTION - Flip sentiment if negation words precede sentiment words
        # Split into sentences for better negation detection
        sentences = re.split(r'[.!?;]', text_lower)

        for sentence in sentences:
            sentence_words = sentence.split()
            for i, word in enumerate(sentence_words):
                # Check if this word is a negation
                if word in self.NEGATION_WORDS:
                    # Look ahead for sentiment words (within next 3 words)
                    for j in range(i+1, min(i+4, len(sentence_words))):
                        next_word = sentence_words[j]
                        # If we find a positive cue, flip it to negative
                        if next_word in self.POSITIVE_CUES:
                            boost -= 0.04  # Penalty for negated positive
                            if debug:
                                debug_info['negations'].append(f"'{word} {next_word}' (-0.04)")
                        # If we find a negative cue, flip it to positive
                        elif next_word in self.NEGATIVE_CUES:
                            boost += 0.04  # Bonus for negated negative
                            if debug:
                                debug_info['negations'].append(f"'{word} {next_word}' (+0.04)")

        # 4. EMPHASIS INDICATORS
        # Exclamation marks
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            emphasis_boost = min(exclamation_count, 3) * 0.02
            boost += emphasis_boost
            if debug:
                debug_info['emphasis']['exclamations'] = f"{exclamation_count} (!!! → +{emphasis_boost:.3f})"

        # ALL CAPS WORDS (indicates strong emotion)
        caps_words = re.findall(r'\b[A-Z]{4,}\b', text)  # 4+ letter all-caps words
        if caps_words:
            caps_boost = min(len(caps_words), 3) * 0.015
            boost += caps_boost
            if debug:
                debug_info['emphasis']['caps_words'] = f"{caps_words} (+{caps_boost:.3f})"

        # Repeated letters (e.g., "sooooo bullish")
        repeated_letter_matches = re.findall(r'(\w)\1{2,}', text_lower)
        if repeated_letter_matches:
            repeat_boost = min(len(repeated_letter_matches), 3) * 0.01
            boost += repeat_boost
            if debug:
                debug_info['emphasis']['repeated_letters'] = f"{len(repeated_letter_matches)} matches (+{repeat_boost:.3f})"

        # 5. EMOJI SENTIMENT
        positive_emojis = ['🚀', '🌙', '💎', '📈', '🔥', '💰', '🤑', '🙌', '💪', '👍', '✅', '🟢']
        negative_emojis = ['📉', '💀', '🔴', '⚠️', '🩸', '👎', '❌', '🔻', '📄']

        emoji_boost = 0
        for emoji in positive_emojis:
            count = text.count(emoji)
            if count > 0:
                emoji_boost += count * 0.02
                if debug:
                    debug_info['emphasis'][f'emoji_{emoji}'] = f"+{count * 0.02:.3f}"

        for emoji in negative_emojis:
            count = text.count(emoji)
            if count > 0:
                emoji_boost -= count * 0.02
                if debug:
                    debug_info['emphasis'][f'emoji_{emoji}'] = f"{count * 0.02:.3f}"

        boost += emoji_boost

        # Cap the boost to prevent extreme swings
        boost_before_cap = boost
        boost = max(-0.25, min(0.25, boost))

        # Combine base VADER score with our custom boost
        score = max(-1.0, min(1.0, base + boost))

        if debug:
            print(f"\n--- SENTIMENT BREAKDOWN ---")
            print(f"Positive words found: {debug_info['positive_words'] or 'None'}")
            print(f"Negative words found: {debug_info['negative_words'] or 'None'}")
            print(f"Positive patterns: {debug_info['positive_patterns'] or 'None'}")
            print(f"Negative patterns: {debug_info['negative_patterns'] or 'None'}")
            print(f"Negations detected: {debug_info['negations'] or 'None'}")
            print(f"Emphasis: {debug_info['emphasis'] or 'None'}")
            print(f"\nBoost calculation: {boost_before_cap:.4f} (capped to {boost:.4f})")
            print(f"Final score: {base:.4f} (base) + {boost:.4f} (boost) = {score:.4f}")
            print(f"{'='*80}\n")

            # Return both score and debug info
            return score, {
                'text_preview': text[:200] + ('...' if len(text) > 200 else ''),
                'base_score': round(base, 4),
                'boost': round(boost, 4),
                'boost_before_cap': round(boost_before_cap, 4),
                'final_score': round(score, 4),
                'positive_words': debug_info['positive_words'],
                'negative_words': debug_info['negative_words'],
                'positive_patterns': debug_info['positive_patterns'],
                'negative_patterns': debug_info['negative_patterns'],
                'negations': debug_info['negations'],
                'emphasis': debug_info['emphasis'],
                'high_confidence': debug_info['high_confidence']
            }

        return score

    def _score_ticker_sentiment(self, ticker: str, title: str, body: str, analyzer, advanced: bool = True, debug: bool = False):
        """
        Score sentiment for a specific ticker using its sentence-level context,
        with a fallback to whole-text scoring.

        Args:
            ticker: Stock ticker symbol
            title: Post title
            body: Post body/content
            analyzer: VADER sentiment analyzer instance
            advanced: Use advanced sentiment features
            debug: If True, return tuple of (score, debug_info), else just score

        Returns:
            Sentiment score between -1.0 and 1.0, or (score, debug_info) if debug=True
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

        if debug:
            print(f"\n🎯 Analyzing ticker: {ticker}")
            print(f"Contexts found: {len(contexts)}")

        # Handle debug mode where score_text returns (score, debug_info)
        results = [self._score_text_with_cues(ctx, analyzer, advanced, debug) for ctx in contexts if ctx]
        if not results:
            return self._score_text_with_cues(combined, analyzer, advanced, debug)

        # Extract scores and debug info if in debug mode
        if debug:
            scores = [r[0] for r in results]
            debug_infos = [r[1] for r in results]
        else:
            scores = results

        avg_score = sum(scores) / len(scores)

        # CONTEXT-AWARE BOOST: Amplify sentiment if ticker appears near outcome indicators
        combined_lower = combined.lower()
        tokens = re.findall(r"[A-Za-z']+", combined_lower)

        positive_outcome_count = sum(1 for token in tokens if token in self.POSITIVE_OUTCOMES)
        negative_outcome_count = sum(1 for token in tokens if token in self.NEGATIVE_OUTCOMES)

        # Apply context multiplier if we detect outcome words
        context_multiplier = 1.0
        if positive_outcome_count > 0 and avg_score > 0:
            # Amplify positive sentiment when positive outcomes are mentioned
            context_multiplier = 1.0 + min(positive_outcome_count * 0.15, 0.3)
            if debug:
                print(f"[Context boost] Found {positive_outcome_count} positive outcome words → {context_multiplier:.2f}x multiplier")
        elif negative_outcome_count > 0 and avg_score < 0:
            # Amplify negative sentiment when negative outcomes are mentioned
            context_multiplier = 1.0 + min(negative_outcome_count * 0.15, 0.3)
            if debug:
                print(f"[Context boost] Found {negative_outcome_count} negative outcome words → {context_multiplier:.2f}x multiplier")

        final_score = max(-1.0, min(1.0, avg_score * context_multiplier))

        if debug:
            print(f"Base sentiment for {ticker}: {avg_score:.4f}")
            print(f"Context multiplier: {context_multiplier:.2f}")
            print(f"Final sentiment: {final_score:.4f}\n")

            # Combine all debug info and return with score
            combined_debug = {
                'ticker': ticker,
                'contexts_found': len(contexts),
                'avg_score': round(avg_score, 4),
                'context_multiplier': round(context_multiplier, 2),
                'final_score': round(final_score, 4),
                'positive_outcome_count': positive_outcome_count,
                'negative_outcome_count': negative_outcome_count,
                'context_details': debug_infos  # All context-level debug info
            }
            return final_score, combined_debug

        return final_score

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
