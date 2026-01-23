"""
Quantsploit Live Trading Module

This module provides live trading capabilities including:
- Real-time data feeds (Alpaca, Polygon, Mock for testing)
- Data normalization to common Quote, Trade, Bar formats
- Failover support via DataFeedManager
- Broker integration for order execution
- Position and risk management
- Event-driven execution engine

Setup:
------
Configure API keys as environment variables:
    - ALPACA_API_KEY: Your Alpaca API key
    - ALPACA_SECRET_KEY: Your Alpaca secret key
    - ALPACA_PAPER: "true" for paper trading (default)
    - POLYGON_API_KEY: Your Polygon.io API key

Quick Start:
------------
>>> from quantsploit.live import AlpacaDataFeed, DataFeedManager
>>>
>>> # Single feed usage
>>> feed = AlpacaDataFeed(paper=True)
>>> await feed.connect()
>>> await feed.subscribe(['AAPL', 'GOOGL'])
>>> feed.on_quote(lambda q: print(f"Quote: {q.symbol} ${q.mid_price:.2f}"))
>>>
>>> # Multi-feed with failover
>>> manager = DataFeedManager()
>>> manager.add_feed('alpaca', AlpacaDataFeed(), priority=1)
>>> manager.add_feed('mock', MockDataFeed(), priority=99)  # Fallback
>>> await manager.start()
"""

from .realtime_data import (
    # Core interface
    DataFeedInterface,
    # Implementations
    AlpacaDataFeed,
    PolygonDataFeed,
    MockDataFeed,
    # Manager
    DataFeedManager,
    # Data structures
    Quote,
    Trade,
    Bar,
    # Configuration
    DataFeedConfig,
    DataFeedStatus,
    DataType,
)

# Broker interface imports
from .broker_interface import (
    # Enums
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    TradingMode,
    PositionSide,

    # Data classes
    Order,
    Position,
    Account,
    RiskLimits,
    BrokerConfig,

    # Core classes
    BrokerInterface,
    AlpacaBroker,
    OrderManager,
    RiskManager,
    PositionReconciler,
    RateLimiter,

    # Exceptions
    BrokerError,
    BrokerConnectionError,
    OrderValidationError,
    OrderNotFoundError,
    OrderModificationError,
    RiskLimitExceededError,
    InsufficientFundsError,
    LiveTradingNotConfirmedError,

    # Factory functions
    create_paper_broker,
    create_live_broker,
)
_BROKER_AVAILABLE = True

__all__ = [
    # Data Feed Interface
    'DataFeedInterface',
    'AlpacaDataFeed',
    'PolygonDataFeed',
    'MockDataFeed',
    'DataFeedManager',
    # Data Structures
    'Quote',
    'Trade',
    'Bar',
    # Configuration
    'DataFeedConfig',
    'DataFeedStatus',
    'DataType',

    # Broker Interface - Enums
    'OrderSide',
    'OrderType',
    'TimeInForce',
    'OrderStatus',
    'TradingMode',
    'PositionSide',

    # Broker Interface - Data classes
    'Order',
    'Position',
    'Account',
    'RiskLimits',
    'BrokerConfig',

    # Broker Interface - Core classes
    'BrokerInterface',
    'AlpacaBroker',
    'OrderManager',
    'RiskManager',
    'PositionReconciler',
    'RateLimiter',

    # Broker Interface - Exceptions
    'BrokerError',
    'BrokerConnectionError',
    'OrderValidationError',
    'OrderNotFoundError',
    'OrderModificationError',
    'RiskLimitExceededError',
    'InsufficientFundsError',
    'LiveTradingNotConfirmedError',

    # Broker Interface - Factory functions
    'create_paper_broker',
    'create_live_broker',
]
