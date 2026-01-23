"""
Real-Time Data Feed Interface for Quantsploit

This module provides abstract and concrete implementations for connecting to
real-time market data providers including Alpaca, Polygon.io, and a mock
data feed for testing purposes.

Setup Instructions:
-------------------
1. Install required dependencies:
   pip install websockets aiohttp pandas yfinance

2. Configure API keys as environment variables:
   - ALPACA_API_KEY: Your Alpaca API key
   - ALPACA_SECRET_KEY: Your Alpaca secret key
   - ALPACA_PAPER: Set to "true" for paper trading (default), "false" for live
   - POLYGON_API_KEY: Your Polygon.io API key

3. Usage example:
   >>> from quantsploit.live import AlpacaDataFeed, DataFeedManager
   >>>
   >>> async def main():
   ...     feed = AlpacaDataFeed(paper=True)
   ...     await feed.connect()
   ...     await feed.subscribe(['AAPL', 'GOOGL'], data_types=['quotes', 'trades'])
   ...
   ...     # Register callbacks
   ...     feed.on_quote(lambda q: print(f"Quote: {q}"))
   ...     feed.on_trade(lambda t: print(f"Trade: {t}"))
   ...
   ...     # Or use DataFeedManager for failover
   ...     manager = DataFeedManager()
   ...     manager.add_feed('alpaca', feed, priority=1)
   ...     await manager.start()

API Documentation:
------------------
- Alpaca: https://alpaca.markets/docs/api-documentation/api-v2/market-data/
- Polygon: https://polygon.io/docs/stocks/getting-started

Note: API keys shown in examples are placeholders. Replace with your actual keys.
"""

import asyncio
import json
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None
    ConnectionClosed = Exception

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

# Configure logging
logger = logging.getLogger(__name__)


class DataFeedStatus(Enum):
    """Status of a data feed connection."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    PAUSED = "paused"


class DataType(Enum):
    """Types of market data available."""
    QUOTE = "quote"
    TRADE = "trade"
    BAR = "bar"
    AGGREGATE = "aggregate"


@dataclass
class DataFeedConfig:
    """
    Configuration for a data feed connection.

    Attributes:
        api_key: API key for authentication
        secret_key: Secret key for authentication (if required)
        base_url: Base URL for REST API
        ws_url: WebSocket URL for streaming data
        paper: Whether to use paper trading mode
        reconnect_attempts: Number of reconnection attempts before giving up
        reconnect_delay: Initial delay between reconnection attempts (seconds)
        max_reconnect_delay: Maximum delay between reconnection attempts
        heartbeat_interval: Interval for sending heartbeat messages (seconds)
    """
    api_key: str = ""
    secret_key: str = ""
    base_url: str = ""
    ws_url: str = ""
    paper: bool = True
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    heartbeat_interval: float = 30.0


@dataclass
class Quote:
    """
    Normalized quote data structure.

    Attributes:
        symbol: Ticker symbol
        bid_price: Current bid price
        bid_size: Bid size in shares
        ask_price: Current ask price
        ask_size: Ask size in shares
        timestamp: Quote timestamp
        exchange: Exchange code
        conditions: List of quote conditions
        tape: Market tape (A, B, or C)
    """
    symbol: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    timestamp: datetime
    exchange: str = ""
    conditions: List[str] = field(default_factory=list)
    tape: str = ""

    @property
    def mid_price(self) -> float:
        """Calculate the mid price."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> float:
        """Calculate the bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_pct(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid = self.mid_price
        return (self.spread / mid * 100) if mid > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'bid_price': self.bid_price,
            'bid_size': self.bid_size,
            'ask_price': self.ask_price,
            'ask_size': self.ask_size,
            'timestamp': self.timestamp.isoformat(),
            'exchange': self.exchange,
            'mid_price': self.mid_price,
            'spread': self.spread,
        }


@dataclass
class Trade:
    """
    Normalized trade data structure.

    Attributes:
        symbol: Ticker symbol
        price: Trade price
        size: Trade size in shares
        timestamp: Trade timestamp
        exchange: Exchange code
        conditions: List of trade conditions
        trade_id: Unique trade identifier
        tape: Market tape (A, B, or C)
    """
    symbol: str
    price: float
    size: int
    timestamp: datetime
    exchange: str = ""
    conditions: List[str] = field(default_factory=list)
    trade_id: str = ""
    tape: str = ""

    @property
    def value(self) -> float:
        """Calculate the trade value (price * size)."""
        return self.price * self.size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'price': self.price,
            'size': self.size,
            'timestamp': self.timestamp.isoformat(),
            'exchange': self.exchange,
            'trade_id': self.trade_id,
            'value': self.value,
        }


@dataclass
class Bar:
    """
    Normalized OHLCV bar data structure.

    Attributes:
        symbol: Ticker symbol
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Volume
        timestamp: Bar timestamp (start of period)
        vwap: Volume-weighted average price
        trade_count: Number of trades in the bar
        timeframe: Bar timeframe (e.g., '1Min', '5Min', '1D')
    """
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    vwap: float = 0.0
    trade_count: int = 0
    timeframe: str = "1Min"

    @property
    def change(self) -> float:
        """Calculate price change."""
        return self.close - self.open

    @property
    def change_pct(self) -> float:
        """Calculate percentage change."""
        return (self.change / self.open * 100) if self.open > 0 else 0.0

    @property
    def range(self) -> float:
        """Calculate high-low range."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Calculate candle body size (absolute value)."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """Check if the bar is bullish (close > open)."""
        return self.close > self.open

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timestamp': self.timestamp.isoformat(),
            'vwap': self.vwap,
            'trade_count': self.trade_count,
            'timeframe': self.timeframe,
            'change_pct': self.change_pct,
        }


class DataFeedInterface(ABC):
    """
    Abstract base class for real-time market data feeds.

    This interface defines the contract that all data feed implementations
    must follow, enabling consistent usage across different providers.

    Implementations should handle:
    - Connection management (connect, disconnect, reconnect)
    - Subscription management (subscribe, unsubscribe)
    - Data retrieval (latest quotes, trades, bars)
    - Callback-based updates for streaming data

    Example:
        >>> class MyDataFeed(DataFeedInterface):
        ...     async def connect(self):
        ...         # Implementation here
        ...         pass
    """

    def __init__(self, config: Optional[DataFeedConfig] = None):
        """
        Initialize the data feed.

        Args:
            config: Optional configuration object. If not provided,
                    default configuration will be used.
        """
        self.config = config or DataFeedConfig()
        self._status = DataFeedStatus.DISCONNECTED
        self._subscriptions: Set[str] = set()
        self._quote_callbacks: List[Callable[[Quote], None]] = []
        self._trade_callbacks: List[Callable[[Trade], None]] = []
        self._bar_callbacks: List[Callable[[Bar], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []
        self._status_callbacks: List[Callable[[DataFeedStatus], None]] = []
        self._latest_quotes: Dict[str, Quote] = {}
        self._latest_trades: Dict[str, Trade] = {}
        self._latest_bars: Dict[str, Bar] = {}
        self._lock = asyncio.Lock()

    @property
    def status(self) -> DataFeedStatus:
        """Get the current connection status."""
        return self._status

    @status.setter
    def status(self, value: DataFeedStatus):
        """Set the connection status and notify callbacks."""
        self._status = value
        for callback in self._status_callbacks:
            try:
                callback(value)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    @property
    def subscriptions(self) -> Set[str]:
        """Get the set of currently subscribed symbols."""
        return self._subscriptions.copy()

    @property
    def is_connected(self) -> bool:
        """Check if the feed is currently connected."""
        return self._status == DataFeedStatus.CONNECTED

    # Connection management

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the data feed.

        Returns:
            True if connection was successful, False otherwise.

        Raises:
            ConnectionError: If connection cannot be established.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the data feed.

        Returns:
            True if disconnection was successful, False otherwise.
        """
        pass

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to the data feed.

        Implements exponential backoff for reconnection attempts.

        Returns:
            True if reconnection was successful, False otherwise.
        """
        self.status = DataFeedStatus.RECONNECTING
        delay = self.config.reconnect_delay

        for attempt in range(self.config.reconnect_attempts):
            logger.info(f"Reconnection attempt {attempt + 1}/{self.config.reconnect_attempts}")

            try:
                await self.disconnect()
                await asyncio.sleep(delay)

                if await self.connect():
                    # Re-subscribe to previous subscriptions
                    if self._subscriptions:
                        await self.subscribe(list(self._subscriptions))
                    return True

            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")

            # Exponential backoff with jitter
            delay = min(delay * 2 + random.uniform(0, 1), self.config.max_reconnect_delay)

        self.status = DataFeedStatus.ERROR
        return False

    # Subscription management

    @abstractmethod
    async def subscribe(
        self,
        symbols: List[str],
        data_types: Optional[List[str]] = None
    ) -> bool:
        """
        Subscribe to real-time data for the given symbols.

        Args:
            symbols: List of ticker symbols to subscribe to.
            data_types: Optional list of data types to subscribe to
                       (e.g., ['quotes', 'trades', 'bars']).
                       If None, subscribes to all available types.

        Returns:
            True if subscription was successful, False otherwise.
        """
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from real-time data for the given symbols.

        Args:
            symbols: List of ticker symbols to unsubscribe from.

        Returns:
            True if unsubscription was successful, False otherwise.
        """
        pass

    # Data retrieval

    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get the latest quote for a symbol.

        Args:
            symbol: The ticker symbol.

        Returns:
            The latest Quote object, or None if not available.
        """
        return self._latest_quotes.get(symbol.upper())

    def get_latest_trade(self, symbol: str) -> Optional[Trade]:
        """
        Get the latest trade for a symbol.

        Args:
            symbol: The ticker symbol.

        Returns:
            The latest Trade object, or None if not available.
        """
        return self._latest_trades.get(symbol.upper())

    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """
        Get the latest bar for a symbol.

        Args:
            symbol: The ticker symbol.

        Returns:
            The latest Bar object, or None if not available.
        """
        return self._latest_bars.get(symbol.upper())

    @abstractmethod
    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Bar]:
        """
        Get historical bar data for a symbol.

        Args:
            symbol: The ticker symbol.
            timeframe: Bar timeframe (e.g., '1Min', '5Min', '1Hour', '1Day').
            start: Start datetime for historical data.
            end: End datetime for historical data.
            limit: Maximum number of bars to retrieve.

        Returns:
            List of Bar objects.
        """
        pass

    # Callback registration

    def on_quote(self, callback: Callable[[Quote], None]) -> None:
        """
        Register a callback for quote updates.

        Args:
            callback: Function to call when a new quote is received.
                     Function signature: callback(quote: Quote) -> None
        """
        self._quote_callbacks.append(callback)

    def on_trade(self, callback: Callable[[Trade], None]) -> None:
        """
        Register a callback for trade updates.

        Args:
            callback: Function to call when a new trade is received.
                     Function signature: callback(trade: Trade) -> None
        """
        self._trade_callbacks.append(callback)

    def on_bar(self, callback: Callable[[Bar], None]) -> None:
        """
        Register a callback for bar updates.

        Args:
            callback: Function to call when a new bar is received.
                     Function signature: callback(bar: Bar) -> None
        """
        self._bar_callbacks.append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """
        Register a callback for error events.

        Args:
            callback: Function to call when an error occurs.
                     Function signature: callback(error: Exception) -> None
        """
        self._error_callbacks.append(callback)

    def on_status_change(self, callback: Callable[[DataFeedStatus], None]) -> None:
        """
        Register a callback for status changes.

        Args:
            callback: Function to call when status changes.
                     Function signature: callback(status: DataFeedStatus) -> None
        """
        self._status_callbacks.append(callback)

    # Internal callback dispatch

    def _dispatch_quote(self, quote: Quote) -> None:
        """Dispatch a quote to all registered callbacks."""
        self._latest_quotes[quote.symbol] = quote
        for callback in self._quote_callbacks:
            try:
                callback(quote)
            except Exception as e:
                logger.error(f"Error in quote callback: {e}")

    def _dispatch_trade(self, trade: Trade) -> None:
        """Dispatch a trade to all registered callbacks."""
        self._latest_trades[trade.symbol] = trade
        for callback in self._trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")

    def _dispatch_bar(self, bar: Bar) -> None:
        """Dispatch a bar to all registered callbacks."""
        self._latest_bars[bar.symbol] = bar
        for callback in self._bar_callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Error in bar callback: {e}")

    def _dispatch_error(self, error: Exception) -> None:
        """Dispatch an error to all registered callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")


class AlpacaDataFeed(DataFeedInterface):
    """
    Alpaca Markets real-time data feed implementation.

    Connects to Alpaca's WebSocket API for real-time quotes, trades, and bars.
    Supports both paper and live trading modes.

    Setup:
        1. Create an Alpaca account at https://alpaca.markets
        2. Generate API keys in the dashboard
        3. Set environment variables:
           - ALPACA_API_KEY: Your API key
           - ALPACA_SECRET_KEY: Your secret key
           - ALPACA_PAPER: "true" for paper trading (default)

    Example:
        >>> feed = AlpacaDataFeed(paper=True)
        >>> await feed.connect()
        >>> await feed.subscribe(['AAPL', 'GOOGL'])
        >>> feed.on_quote(lambda q: print(f"Quote: {q.symbol} ${q.mid_price:.2f}"))

    API Documentation:
        https://alpaca.markets/docs/api-documentation/api-v2/market-data/streaming/
    """

    # Alpaca WebSocket URLs
    PAPER_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"
    LIVE_WS_URL = "wss://stream.data.alpaca.markets/v2/sip"
    PAPER_BASE_URL = "https://paper-api.alpaca.markets"
    LIVE_BASE_URL = "https://api.alpaca.markets"
    DATA_BASE_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        config: Optional[DataFeedConfig] = None
    ):
        """
        Initialize Alpaca data feed.

        Args:
            api_key: Alpaca API key. If not provided, reads from ALPACA_API_KEY env var.
            secret_key: Alpaca secret key. If not provided, reads from ALPACA_SECRET_KEY env var.
            paper: Whether to use paper trading mode (default True).
            config: Optional DataFeedConfig for additional settings.
        """
        super().__init__(config)

        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "YOUR_ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")
        self.paper = paper if paper is not None else os.environ.get("ALPACA_PAPER", "true").lower() == "true"

        self.config.ws_url = self.PAPER_WS_URL if self.paper else self.LIVE_WS_URL
        self.config.base_url = self.PAPER_BASE_URL if self.paper else self.LIVE_BASE_URL

        self._ws: Optional[Any] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._authenticated = False

    async def connect(self) -> bool:
        """
        Connect to Alpaca WebSocket API.

        Returns:
            True if connection and authentication were successful.

        Raises:
            ImportError: If websockets library is not installed.
            ConnectionError: If connection fails.
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets library is required for AlpacaDataFeed. "
                "Install it with: pip install websockets"
            )

        self.status = DataFeedStatus.CONNECTING
        logger.info(f"Connecting to Alpaca {'paper' if self.paper else 'live'} data feed...")

        try:
            self._ws = await websockets.connect(self.config.ws_url)

            # Wait for welcome message
            welcome = await self._ws.recv()
            welcome_data = json.loads(welcome)
            logger.debug(f"Received welcome: {welcome_data}")

            # Authenticate
            auth_message = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }
            await self._ws.send(json.dumps(auth_message))

            # Wait for auth response
            auth_response = await self._ws.recv()
            auth_data = json.loads(auth_response)
            logger.debug(f"Auth response: {auth_data}")

            if isinstance(auth_data, list) and len(auth_data) > 0:
                if auth_data[0].get("T") == "success" and auth_data[0].get("msg") == "authenticated":
                    self._authenticated = True
                    self.status = DataFeedStatus.CONNECTED
                    logger.info("Successfully connected and authenticated to Alpaca")

                    # Start message handler
                    self._ws_task = asyncio.create_task(self._message_handler())

                    # Start heartbeat
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                    return True

            logger.error(f"Authentication failed: {auth_data}")
            self.status = DataFeedStatus.ERROR
            return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.status = DataFeedStatus.ERROR
            self._dispatch_error(e)
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from Alpaca WebSocket API.

        Returns:
            True if disconnection was successful.
        """
        logger.info("Disconnecting from Alpaca data feed...")

        # Cancel tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

        # Close WebSocket
        if self._ws:
            await self._ws.close()
            self._ws = None

        self._authenticated = False
        self.status = DataFeedStatus.DISCONNECTED
        logger.info("Disconnected from Alpaca data feed")
        return True

    async def subscribe(
        self,
        symbols: List[str],
        data_types: Optional[List[str]] = None
    ) -> bool:
        """
        Subscribe to real-time data for symbols.

        Args:
            symbols: List of ticker symbols.
            data_types: List of data types ('quotes', 'trades', 'bars').
                       Defaults to all types.

        Returns:
            True if subscription was successful.
        """
        if not self.is_connected:
            logger.error("Cannot subscribe: not connected")
            return False

        data_types = data_types or ['quotes', 'trades', 'bars']
        symbols = [s.upper() for s in symbols]

        subscribe_message = {"action": "subscribe"}

        if 'quotes' in data_types:
            subscribe_message["quotes"] = symbols
        if 'trades' in data_types:
            subscribe_message["trades"] = symbols
        if 'bars' in data_types:
            subscribe_message["bars"] = symbols

        try:
            await self._ws.send(json.dumps(subscribe_message))
            self._subscriptions.update(symbols)
            logger.info(f"Subscribed to {symbols} for {data_types}")
            return True
        except Exception as e:
            logger.error(f"Subscribe error: {e}")
            self._dispatch_error(e)
            return False

    async def unsubscribe(self, symbols: List[str]) -> bool:
        """
        Unsubscribe from symbols.

        Args:
            symbols: List of ticker symbols to unsubscribe from.

        Returns:
            True if unsubscription was successful.
        """
        if not self.is_connected:
            return False

        symbols = [s.upper() for s in symbols]

        unsubscribe_message = {
            "action": "unsubscribe",
            "quotes": symbols,
            "trades": symbols,
            "bars": symbols
        }

        try:
            await self._ws.send(json.dumps(unsubscribe_message))
            self._subscriptions.difference_update(symbols)
            logger.info(f"Unsubscribed from {symbols}")
            return True
        except Exception as e:
            logger.error(f"Unsubscribe error: {e}")
            return False

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Bar]:
        """
        Get historical bar data from Alpaca.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day).
            start: Start datetime.
            end: End datetime.
            limit: Maximum bars to retrieve.

        Returns:
            List of Bar objects.
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required for historical data. Install with: pip install aiohttp")

        # Default to last 30 days if no dates provided
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=30)

        url = f"{self.DATA_BASE_URL}/v2/stocks/{symbol}/bars"
        params = {
            "timeframe": timeframe,
            "start": start.isoformat() + "Z",
            "end": end.isoformat() + "Z",
            "limit": limit
        }

        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }

        bars = []

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    for bar_data in data.get("bars", []):
                        bars.append(Bar(
                            symbol=symbol.upper(),
                            open=bar_data["o"],
                            high=bar_data["h"],
                            low=bar_data["l"],
                            close=bar_data["c"],
                            volume=bar_data["v"],
                            timestamp=datetime.fromisoformat(bar_data["t"].replace("Z", "+00:00")),
                            vwap=bar_data.get("vw", 0.0),
                            trade_count=bar_data.get("n", 0),
                            timeframe=timeframe
                        ))
                else:
                    logger.error(f"Failed to get historical bars: {response.status}")

        return bars

    async def _message_handler(self) -> None:
        """Handle incoming WebSocket messages."""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    await self._process_messages(data)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
            if self.status == DataFeedStatus.CONNECTED:
                await self.reconnect()
        except asyncio.CancelledError:
            logger.debug("Message handler cancelled")
            raise
        except Exception as e:
            logger.error(f"Message handler error: {e}")
            self._dispatch_error(e)

    async def _process_messages(self, messages: List[Dict]) -> None:
        """Process a list of messages from Alpaca."""
        if not isinstance(messages, list):
            messages = [messages]

        for msg in messages:
            msg_type = msg.get("T")

            if msg_type == "q":  # Quote
                quote = Quote(
                    symbol=msg["S"],
                    bid_price=msg.get("bp", 0.0),
                    bid_size=msg.get("bs", 0),
                    ask_price=msg.get("ap", 0.0),
                    ask_size=msg.get("as", 0),
                    timestamp=datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                    exchange=msg.get("bx", ""),
                    conditions=msg.get("c", []),
                    tape=msg.get("z", "")
                )
                self._dispatch_quote(quote)

            elif msg_type == "t":  # Trade
                trade = Trade(
                    symbol=msg["S"],
                    price=msg["p"],
                    size=msg["s"],
                    timestamp=datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                    exchange=msg.get("x", ""),
                    conditions=msg.get("c", []),
                    trade_id=str(msg.get("i", "")),
                    tape=msg.get("z", "")
                )
                self._dispatch_trade(trade)

            elif msg_type == "b":  # Bar
                bar = Bar(
                    symbol=msg["S"],
                    open=msg["o"],
                    high=msg["h"],
                    low=msg["l"],
                    close=msg["c"],
                    volume=msg["v"],
                    timestamp=datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                    vwap=msg.get("vw", 0.0),
                    trade_count=msg.get("n", 0),
                    timeframe="1Min"
                )
                self._dispatch_bar(bar)

            elif msg_type == "subscription":
                logger.debug(f"Subscription update: {msg}")

            elif msg_type == "error":
                logger.error(f"Alpaca error: {msg.get('msg')}")
                self._dispatch_error(Exception(msg.get('msg', 'Unknown error')))

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to keep connection alive."""
        try:
            while self.is_connected:
                await asyncio.sleep(self.config.heartbeat_interval)
                # Alpaca doesn't require explicit heartbeat, connection is maintained
                # This is just for monitoring
                logger.debug("Heartbeat: connection alive")
        except asyncio.CancelledError:
            logger.debug("Heartbeat loop cancelled")
            raise


class PolygonDataFeed(DataFeedInterface):
    """
    Polygon.io real-time data feed implementation.

    Connects to Polygon's WebSocket API for real-time quotes, trades, and aggregates.
    Supports stocks, options, forex, and crypto.

    Setup:
        1. Create a Polygon account at https://polygon.io
        2. Get your API key from the dashboard
        3. Set environment variable: POLYGON_API_KEY=your_key

    Example:
        >>> feed = PolygonDataFeed()
        >>> await feed.connect()
        >>> await feed.subscribe(['AAPL', 'GOOGL'], data_types=['quotes', 'trades'])
        >>> feed.on_trade(lambda t: print(f"Trade: {t.symbol} @ ${t.price}"))

    API Documentation:
        https://polygon.io/docs/stocks/ws_stocks_am

    Note: Polygon requires different subscriptions for different data types:
        - Stocks: Q.* (quotes), T.* (trades), A.* (second aggs), AM.* (minute aggs)
        - Options: Q.* (quotes), T.* (trades)
        - Crypto: XQ.* (quotes), XT.* (trades), XA.* (aggs)
    """

    # Polygon WebSocket URLs
    STOCKS_WS_URL = "wss://socket.polygon.io/stocks"
    OPTIONS_WS_URL = "wss://socket.polygon.io/options"
    FOREX_WS_URL = "wss://socket.polygon.io/forex"
    CRYPTO_WS_URL = "wss://socket.polygon.io/crypto"
    REST_BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: Optional[str] = None,
        market: str = "stocks",
        config: Optional[DataFeedConfig] = None
    ):
        """
        Initialize Polygon data feed.

        Args:
            api_key: Polygon API key. If not provided, reads from POLYGON_API_KEY env var.
            market: Market type ('stocks', 'options', 'forex', 'crypto').
            config: Optional DataFeedConfig for additional settings.
        """
        super().__init__(config)

        self.api_key = api_key or os.environ.get("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY")
        self.market = market.lower()

        # Set WebSocket URL based on market
        ws_urls = {
            "stocks": self.STOCKS_WS_URL,
            "options": self.OPTIONS_WS_URL,
            "forex": self.FOREX_WS_URL,
            "crypto": self.CRYPTO_WS_URL
        }
        self.config.ws_url = ws_urls.get(self.market, self.STOCKS_WS_URL)
        self.config.base_url = self.REST_BASE_URL

        self._ws: Optional[Any] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._authenticated = False

    async def connect(self) -> bool:
        """
        Connect to Polygon WebSocket API.

        Returns:
            True if connection and authentication were successful.
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets library is required. Install with: pip install websockets"
            )

        self.status = DataFeedStatus.CONNECTING
        logger.info(f"Connecting to Polygon {self.market} data feed...")

        try:
            self._ws = await websockets.connect(self.config.ws_url)

            # Wait for connection message
            connected = await self._ws.recv()
            connected_data = json.loads(connected)
            logger.debug(f"Connection response: {connected_data}")

            # Authenticate
            auth_message = {"action": "auth", "params": self.api_key}
            await self._ws.send(json.dumps(auth_message))

            # Wait for auth response
            auth_response = await self._ws.recv()
            auth_data = json.loads(auth_response)
            logger.debug(f"Auth response: {auth_data}")

            if isinstance(auth_data, list) and len(auth_data) > 0:
                if auth_data[0].get("status") == "auth_success":
                    self._authenticated = True
                    self.status = DataFeedStatus.CONNECTED
                    logger.info("Successfully connected to Polygon")

                    # Start message handler
                    self._ws_task = asyncio.create_task(self._message_handler())

                    return True

            logger.error(f"Polygon authentication failed: {auth_data}")
            self.status = DataFeedStatus.ERROR
            return False

        except Exception as e:
            logger.error(f"Polygon connection error: {e}")
            self.status = DataFeedStatus.ERROR
            self._dispatch_error(e)
            return False

    async def disconnect(self) -> bool:
        """Disconnect from Polygon WebSocket API."""
        logger.info("Disconnecting from Polygon data feed...")

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._authenticated = False
        self.status = DataFeedStatus.DISCONNECTED
        logger.info("Disconnected from Polygon")
        return True

    async def subscribe(
        self,
        symbols: List[str],
        data_types: Optional[List[str]] = None
    ) -> bool:
        """
        Subscribe to real-time data for symbols.

        Args:
            symbols: List of ticker symbols.
            data_types: List of data types ('quotes', 'trades', 'aggregates').

        Returns:
            True if subscription was successful.
        """
        if not self.is_connected:
            logger.error("Cannot subscribe: not connected")
            return False

        data_types = data_types or ['quotes', 'trades', 'aggregates']
        symbols = [s.upper() for s in symbols]

        # Build subscription channels
        channels = []
        for symbol in symbols:
            if 'quotes' in data_types:
                channels.append(f"Q.{symbol}")
            if 'trades' in data_types:
                channels.append(f"T.{symbol}")
            if 'aggregates' in data_types:
                channels.append(f"AM.{symbol}")  # Minute aggregates

        subscribe_message = {
            "action": "subscribe",
            "params": ",".join(channels)
        }

        try:
            await self._ws.send(json.dumps(subscribe_message))
            self._subscriptions.update(symbols)
            logger.info(f"Subscribed to {symbols} for {data_types}")
            return True
        except Exception as e:
            logger.error(f"Subscribe error: {e}")
            self._dispatch_error(e)
            return False

    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols."""
        if not self.is_connected:
            return False

        symbols = [s.upper() for s in symbols]

        # Build unsubscription channels
        channels = []
        for symbol in symbols:
            channels.extend([f"Q.{symbol}", f"T.{symbol}", f"AM.{symbol}"])

        unsubscribe_message = {
            "action": "unsubscribe",
            "params": ",".join(channels)
        }

        try:
            await self._ws.send(json.dumps(unsubscribe_message))
            self._subscriptions.difference_update(symbols)
            logger.info(f"Unsubscribed from {symbols}")
            return True
        except Exception as e:
            logger.error(f"Unsubscribe error: {e}")
            return False

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Bar]:
        """
        Get historical bar data from Polygon.

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe (1, 5, 15 for minutes; or 'hour', 'day', 'week').
            start: Start datetime.
            end: End datetime.
            limit: Maximum bars to retrieve.

        Returns:
            List of Bar objects.
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required. Install with: pip install aiohttp")

        # Parse timeframe
        multiplier = 1
        span = "day"
        if timeframe.endswith("Min"):
            multiplier = int(timeframe.replace("Min", ""))
            span = "minute"
        elif timeframe.endswith("Hour"):
            multiplier = int(timeframe.replace("Hour", "")) if timeframe.replace("Hour", "") else 1
            span = "hour"
        elif timeframe == "1Day":
            multiplier = 1
            span = "day"
        elif timeframe == "1Week":
            multiplier = 1
            span = "week"

        # Default dates
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=30)

        url = f"{self.REST_BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{span}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        params = {
            "apiKey": self.api_key,
            "limit": limit,
            "sort": "asc"
        }

        bars = []

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    for bar_data in data.get("results", []):
                        bars.append(Bar(
                            symbol=symbol.upper(),
                            open=bar_data["o"],
                            high=bar_data["h"],
                            low=bar_data["l"],
                            close=bar_data["c"],
                            volume=bar_data["v"],
                            timestamp=datetime.fromtimestamp(bar_data["t"] / 1000),
                            vwap=bar_data.get("vw", 0.0),
                            trade_count=bar_data.get("n", 0),
                            timeframe=timeframe
                        ))
                else:
                    logger.error(f"Failed to get Polygon historical bars: {response.status}")

        return bars

    async def _message_handler(self) -> None:
        """Handle incoming WebSocket messages."""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    await self._process_messages(data)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
        except ConnectionClosed:
            logger.warning("Polygon WebSocket connection closed")
            if self.status == DataFeedStatus.CONNECTED:
                await self.reconnect()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Polygon message handler error: {e}")
            self._dispatch_error(e)

    async def _process_messages(self, messages: List[Dict]) -> None:
        """Process messages from Polygon."""
        if not isinstance(messages, list):
            messages = [messages]

        for msg in messages:
            ev = msg.get("ev")

            if ev == "Q":  # Quote
                quote = Quote(
                    symbol=msg["sym"],
                    bid_price=msg.get("bp", 0.0),
                    bid_size=msg.get("bs", 0),
                    ask_price=msg.get("ap", 0.0),
                    ask_size=msg.get("as", 0),
                    timestamp=datetime.fromtimestamp(msg["t"] / 1000000000),
                    exchange=str(msg.get("bx", "")),
                    conditions=msg.get("c", [])
                )
                self._dispatch_quote(quote)

            elif ev == "T":  # Trade
                trade = Trade(
                    symbol=msg["sym"],
                    price=msg["p"],
                    size=msg["s"],
                    timestamp=datetime.fromtimestamp(msg["t"] / 1000000000),
                    exchange=str(msg.get("x", "")),
                    conditions=msg.get("c", []),
                    trade_id=str(msg.get("i", ""))
                )
                self._dispatch_trade(trade)

            elif ev == "AM":  # Minute aggregate
                bar = Bar(
                    symbol=msg["sym"],
                    open=msg["o"],
                    high=msg["h"],
                    low=msg["l"],
                    close=msg["c"],
                    volume=msg["v"],
                    timestamp=datetime.fromtimestamp(msg["s"] / 1000),
                    vwap=msg.get("vw", 0.0),
                    trade_count=msg.get("n", 0),
                    timeframe="1Min"
                )
                self._dispatch_bar(bar)

            elif ev == "status":
                logger.debug(f"Polygon status: {msg.get('message')}")


class MockDataFeed(DataFeedInterface):
    """
    Mock data feed for testing and paper trading.

    Simulates real-time market data from historical data or generates
    synthetic data for testing purposes. Useful for:
    - Backtesting strategies with simulated real-time execution
    - Testing trading infrastructure without live connections
    - Development and debugging without API rate limits

    Example:
        >>> feed = MockDataFeed(speed=2.0)  # 2x playback speed
        >>> await feed.connect()
        >>> await feed.subscribe(['AAPL'])
        >>> feed.on_quote(lambda q: print(f"Mock quote: {q}"))
        >>> await feed.start_simulation()

    Attributes:
        speed: Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed)
        use_historical: Whether to use yfinance historical data
        synthetic_volatility: Volatility for synthetic data generation
    """

    def __init__(
        self,
        speed: float = 1.0,
        use_historical: bool = True,
        synthetic_volatility: float = 0.02,
        config: Optional[DataFeedConfig] = None
    ):
        """
        Initialize mock data feed.

        Args:
            speed: Playback speed multiplier.
            use_historical: Whether to fetch and use historical data.
            synthetic_volatility: Daily volatility for synthetic data.
            config: Optional DataFeedConfig.
        """
        super().__init__(config)

        self.speed = speed
        self.use_historical = use_historical
        self.synthetic_volatility = synthetic_volatility

        self._simulation_task: Optional[asyncio.Task] = None
        self._running = False
        self._historical_data: Dict[str, List[Bar]] = {}
        self._current_prices: Dict[str, float] = {}
        self._base_interval = 1.0  # Base interval between updates in seconds

    async def connect(self) -> bool:
        """
        Initialize the mock data feed.

        Returns:
            True always (mock connection always succeeds).
        """
        self.status = DataFeedStatus.CONNECTING
        logger.info("Initializing mock data feed...")

        # Load historical data if available
        if self.use_historical and YFINANCE_AVAILABLE:
            logger.info("Historical data loading enabled (will load on first subscription)")
        elif self.use_historical:
            logger.warning("yfinance not available, using synthetic data only")
            self.use_historical = False

        self.status = DataFeedStatus.CONNECTED
        logger.info("Mock data feed ready")
        return True

    async def disconnect(self) -> bool:
        """Stop the mock data feed."""
        logger.info("Disconnecting mock data feed...")

        await self.stop_simulation()

        self._historical_data.clear()
        self._current_prices.clear()

        self.status = DataFeedStatus.DISCONNECTED
        logger.info("Mock data feed disconnected")
        return True

    async def subscribe(
        self,
        symbols: List[str],
        data_types: Optional[List[str]] = None
    ) -> bool:
        """
        Subscribe to symbols and optionally load historical data.

        Args:
            symbols: List of ticker symbols.
            data_types: Ignored for mock feed (provides all types).

        Returns:
            True if subscription was successful.
        """
        symbols = [s.upper() for s in symbols]

        # Load historical data for new symbols
        if self.use_historical and YFINANCE_AVAILABLE:
            for symbol in symbols:
                if symbol not in self._historical_data:
                    await self._load_historical_data(symbol)

        # Initialize current prices for synthetic data
        for symbol in symbols:
            if symbol not in self._current_prices:
                if symbol in self._historical_data and self._historical_data[symbol]:
                    self._current_prices[symbol] = self._historical_data[symbol][-1].close
                else:
                    # Default starting price
                    self._current_prices[symbol] = 100.0

        self._subscriptions.update(symbols)
        logger.info(f"Mock feed subscribed to {symbols}")
        return True

    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols."""
        symbols = [s.upper() for s in symbols]
        self._subscriptions.difference_update(symbols)

        # Clean up data for unsubscribed symbols
        for symbol in symbols:
            self._historical_data.pop(symbol, None)
            self._current_prices.pop(symbol, None)

        logger.info(f"Mock feed unsubscribed from {symbols}")
        return True

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Bar]:
        """
        Get historical bar data (from yfinance or cached).

        Args:
            symbol: Ticker symbol.
            timeframe: Bar timeframe.
            start: Start datetime.
            end: End datetime.
            limit: Maximum bars.

        Returns:
            List of Bar objects.
        """
        symbol = symbol.upper()

        # Check cache first
        if symbol in self._historical_data:
            bars = self._historical_data[symbol]
            if start:
                bars = [b for b in bars if b.timestamp >= start]
            if end:
                bars = [b for b in bars if b.timestamp <= end]
            return bars[:limit]

        # Fetch from yfinance
        if YFINANCE_AVAILABLE:
            await self._load_historical_data(symbol, start, end)
            return self._historical_data.get(symbol, [])[:limit]

        # Generate synthetic historical data
        return self._generate_synthetic_bars(symbol, limit)

    async def start_simulation(self) -> None:
        """
        Start the data simulation.

        This begins generating mock market data for all subscribed symbols.
        """
        if self._running:
            logger.warning("Simulation already running")
            return

        self._running = True
        self._simulation_task = asyncio.create_task(self._simulation_loop())
        logger.info(f"Started mock data simulation at {self.speed}x speed")

    async def stop_simulation(self) -> None:
        """Stop the data simulation."""
        self._running = False

        if self._simulation_task:
            self._simulation_task.cancel()
            try:
                await self._simulation_task
            except asyncio.CancelledError:
                pass
            self._simulation_task = None

        logger.info("Stopped mock data simulation")

    def set_price(self, symbol: str, price: float) -> None:
        """
        Manually set the current price for a symbol.

        Useful for testing specific price scenarios.

        Args:
            symbol: Ticker symbol.
            price: Price to set.
        """
        self._current_prices[symbol.upper()] = price

    async def _load_historical_data(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> None:
        """Load historical data from yfinance."""
        try:
            ticker = yf.Ticker(symbol)

            # Default to last 30 days
            if end is None:
                end = datetime.now()
            if start is None:
                start = end - timedelta(days=30)

            # Fetch data
            df = ticker.history(start=start, end=end)

            if df.empty:
                logger.warning(f"No historical data available for {symbol}")
                return

            bars = []
            for idx, row in df.iterrows():
                bars.append(Bar(
                    symbol=symbol,
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    volume=int(row['Volume']),
                    timestamp=idx.to_pydatetime(),
                    timeframe="1Day"
                ))

            self._historical_data[symbol] = bars
            logger.info(f"Loaded {len(bars)} historical bars for {symbol}")

        except Exception as e:
            logger.error(f"Failed to load historical data for {symbol}: {e}")

    def _generate_synthetic_bars(self, symbol: str, count: int) -> List[Bar]:
        """Generate synthetic historical bars."""
        bars = []
        price = self._current_prices.get(symbol, 100.0)
        timestamp = datetime.now() - timedelta(days=count)

        for _ in range(count):
            # Random walk with drift
            change = random.gauss(0.0001, self.synthetic_volatility)
            price *= (1 + change)

            high = price * (1 + abs(random.gauss(0, 0.01)))
            low = price * (1 - abs(random.gauss(0, 0.01)))
            open_price = random.uniform(low, high)

            bars.append(Bar(
                symbol=symbol,
                open=open_price,
                high=high,
                low=low,
                close=price,
                volume=random.randint(100000, 10000000),
                timestamp=timestamp,
                timeframe="1Day"
            ))

            timestamp += timedelta(days=1)

        return bars

    async def _simulation_loop(self) -> None:
        """Main simulation loop generating mock data."""
        try:
            while self._running:
                for symbol in list(self._subscriptions):
                    await self._generate_tick(symbol)

                # Wait based on speed setting
                await asyncio.sleep(self._base_interval / self.speed)

        except asyncio.CancelledError:
            logger.debug("Simulation loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Simulation loop error: {e}")
            self._dispatch_error(e)

    async def _generate_tick(self, symbol: str) -> None:
        """Generate a single tick of mock data for a symbol."""
        # Get or initialize current price
        price = self._current_prices.get(symbol, 100.0)

        # Random walk
        change = random.gauss(0, self.synthetic_volatility / 100)
        price *= (1 + change)
        self._current_prices[symbol] = price

        # Generate spread
        spread = price * random.uniform(0.0001, 0.001)

        now = datetime.now()

        # Generate quote
        quote = Quote(
            symbol=symbol,
            bid_price=price - spread / 2,
            bid_size=random.randint(100, 1000),
            ask_price=price + spread / 2,
            ask_size=random.randint(100, 1000),
            timestamp=now,
            exchange="MOCK"
        )
        self._dispatch_quote(quote)

        # Generate trade (not every tick)
        if random.random() < 0.5:
            trade_price = random.uniform(quote.bid_price, quote.ask_price)
            trade = Trade(
                symbol=symbol,
                price=trade_price,
                size=random.randint(1, 500),
                timestamp=now,
                exchange="MOCK",
                trade_id=f"MOCK-{int(time.time() * 1000000)}"
            )
            self._dispatch_trade(trade)


class DataFeedManager:
    """
    Manager for multiple data feed sources with failover support.

    Handles multiple data providers, normalizes data to a common format,
    and provides automatic failover when a primary feed fails.

    Features:
    - Multiple feed management with priority-based failover
    - Data normalization across providers
    - Automatic reconnection and failover
    - Unified callback interface
    - Health monitoring and status reporting

    Example:
        >>> manager = DataFeedManager()
        >>>
        >>> # Add feeds with priorities (lower = higher priority)
        >>> manager.add_feed('alpaca', AlpacaDataFeed(paper=True), priority=1)
        >>> manager.add_feed('polygon', PolygonDataFeed(), priority=2)
        >>> manager.add_feed('mock', MockDataFeed(), priority=99)  # Fallback
        >>>
        >>> # Register callbacks on manager (applies to active feed)
        >>> manager.on_quote(lambda q: print(f"Quote: {q}"))
        >>>
        >>> # Start manager
        >>> await manager.start()
        >>> await manager.subscribe(['AAPL', 'GOOGL'])

    Attributes:
        feeds: Dictionary of registered data feeds
        active_feed: Currently active data feed
        failover_enabled: Whether automatic failover is enabled
    """

    def __init__(self, failover_enabled: bool = True):
        """
        Initialize the data feed manager.

        Args:
            failover_enabled: Whether to automatically fail over to
                            backup feeds when the primary fails.
        """
        self._feeds: Dict[str, Dict[str, Any]] = {}
        self._active_feed_name: Optional[str] = None
        self.failover_enabled = failover_enabled

        # Unified callbacks
        self._quote_callbacks: List[Callable[[Quote], None]] = []
        self._trade_callbacks: List[Callable[[Trade], None]] = []
        self._bar_callbacks: List[Callable[[Bar], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []

        # Subscriptions to maintain across failover
        self._subscriptions: Set[str] = set()
        self._data_types: List[str] = ['quotes', 'trades', 'bars']

        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def feeds(self) -> Dict[str, DataFeedInterface]:
        """Get dictionary of all registered feeds."""
        return {name: info['feed'] for name, info in self._feeds.items()}

    @property
    def active_feed(self) -> Optional[DataFeedInterface]:
        """Get the currently active data feed."""
        if self._active_feed_name:
            return self._feeds[self._active_feed_name]['feed']
        return None

    @property
    def active_feed_name(self) -> Optional[str]:
        """Get the name of the currently active feed."""
        return self._active_feed_name

    def add_feed(
        self,
        name: str,
        feed: DataFeedInterface,
        priority: int = 10
    ) -> None:
        """
        Register a data feed with the manager.

        Args:
            name: Unique identifier for the feed.
            feed: DataFeedInterface implementation.
            priority: Feed priority (lower = higher priority).
        """
        self._feeds[name] = {
            'feed': feed,
            'priority': priority,
            'healthy': False,
            'last_error': None,
            'error_count': 0
        }

        # Wire up callbacks to forward to manager callbacks
        feed.on_quote(self._handle_quote)
        feed.on_trade(self._handle_trade)
        feed.on_bar(self._handle_bar)
        feed.on_error(lambda e: self._handle_error(name, e))
        feed.on_status_change(lambda s: self._handle_status_change(name, s))

        logger.info(f"Added feed '{name}' with priority {priority}")

    def remove_feed(self, name: str) -> bool:
        """
        Remove a data feed from the manager.

        Args:
            name: Name of the feed to remove.

        Returns:
            True if feed was removed, False if not found.
        """
        if name not in self._feeds:
            return False

        if name == self._active_feed_name:
            logger.warning(f"Removing active feed '{name}', will trigger failover")

        del self._feeds[name]
        logger.info(f"Removed feed '{name}'")
        return True

    async def start(self) -> bool:
        """
        Start the data feed manager.

        Connects to the highest priority available feed.

        Returns:
            True if at least one feed connected successfully.
        """
        logger.info("Starting data feed manager...")
        self._running = True

        # Try to connect to feeds in priority order
        connected = await self._connect_best_feed()

        if connected:
            # Start health monitoring
            self._monitor_task = asyncio.create_task(self._health_monitor())
            logger.info(f"Data feed manager started with '{self._active_feed_name}'")
        else:
            logger.error("Failed to connect to any data feed")

        return connected

    async def stop(self) -> None:
        """Stop the data feed manager and disconnect all feeds."""
        logger.info("Stopping data feed manager...")
        self._running = False

        # Stop health monitor
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        # Disconnect all feeds
        for name, info in self._feeds.items():
            try:
                await info['feed'].disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting feed '{name}': {e}")

        self._active_feed_name = None
        logger.info("Data feed manager stopped")

    async def subscribe(
        self,
        symbols: List[str],
        data_types: Optional[List[str]] = None
    ) -> bool:
        """
        Subscribe to symbols through the active feed.

        Args:
            symbols: List of ticker symbols.
            data_types: List of data types to subscribe to.

        Returns:
            True if subscription was successful.
        """
        self._subscriptions.update(s.upper() for s in symbols)
        if data_types:
            self._data_types = data_types

        if self.active_feed:
            return await self.active_feed.subscribe(symbols, data_types)

        logger.warning("No active feed for subscription")
        return False

    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols."""
        self._subscriptions.difference_update(s.upper() for s in symbols)

        if self.active_feed:
            return await self.active_feed.unsubscribe(symbols)

        return False

    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """Get latest quote from the active feed."""
        if self.active_feed:
            return self.active_feed.get_latest_quote(symbol)
        return None

    def get_latest_trade(self, symbol: str) -> Optional[Trade]:
        """Get latest trade from the active feed."""
        if self.active_feed:
            return self.active_feed.get_latest_trade(symbol)
        return None

    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        """Get latest bar from the active feed."""
        if self.active_feed:
            return self.active_feed.get_latest_bar(symbol)
        return None

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Bar]:
        """Get historical bars from the active feed."""
        if self.active_feed:
            return await self.active_feed.get_historical_bars(
                symbol, timeframe, start, end, limit
            )
        return []

    # Callback registration

    def on_quote(self, callback: Callable[[Quote], None]) -> None:
        """Register a callback for quote updates."""
        self._quote_callbacks.append(callback)

    def on_trade(self, callback: Callable[[Trade], None]) -> None:
        """Register a callback for trade updates."""
        self._trade_callbacks.append(callback)

    def on_bar(self, callback: Callable[[Bar], None]) -> None:
        """Register a callback for bar updates."""
        self._bar_callbacks.append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register a callback for errors."""
        self._error_callbacks.append(callback)

    # Status and health

    def get_feed_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all registered feeds.

        Returns:
            Dictionary with feed names as keys and status info as values.
        """
        status = {}
        for name, info in self._feeds.items():
            feed = info['feed']
            status[name] = {
                'status': feed.status.value,
                'connected': feed.is_connected,
                'priority': info['priority'],
                'healthy': info['healthy'],
                'error_count': info['error_count'],
                'last_error': str(info['last_error']) if info['last_error'] else None,
                'subscriptions': list(feed.subscriptions),
                'is_active': name == self._active_feed_name
            }
        return status

    # Internal methods

    async def _connect_best_feed(self) -> bool:
        """Connect to the highest priority available feed."""
        # Sort feeds by priority
        sorted_feeds = sorted(
            self._feeds.items(),
            key=lambda x: x[1]['priority']
        )

        for name, info in sorted_feeds:
            feed = info['feed']

            try:
                logger.info(f"Attempting to connect to '{name}'...")
                if await feed.connect():
                    self._active_feed_name = name
                    info['healthy'] = True
                    info['error_count'] = 0

                    # Restore subscriptions
                    if self._subscriptions:
                        await feed.subscribe(
                            list(self._subscriptions),
                            self._data_types
                        )

                    return True

            except Exception as e:
                logger.error(f"Failed to connect to '{name}': {e}")
                info['last_error'] = e
                info['error_count'] += 1

        return False

    async def _failover(self) -> bool:
        """Attempt to fail over to a backup feed."""
        if not self.failover_enabled:
            logger.warning("Failover disabled, not attempting backup feeds")
            return False

        logger.info("Initiating failover to backup feed...")

        # Mark current feed as unhealthy
        if self._active_feed_name:
            self._feeds[self._active_feed_name]['healthy'] = False

        # Try to connect to next best available feed
        old_active = self._active_feed_name
        self._active_feed_name = None

        if await self._connect_best_feed():
            logger.info(f"Failover successful: {old_active} -> {self._active_feed_name}")
            return True

        # Restore old feed name even though it's unhealthy
        self._active_feed_name = old_active
        logger.error("Failover failed: no backup feeds available")
        return False

    def _handle_quote(self, quote: Quote) -> None:
        """Handle quote from active feed."""
        for callback in self._quote_callbacks:
            try:
                callback(quote)
            except Exception as e:
                logger.error(f"Error in quote callback: {e}")

    def _handle_trade(self, trade: Trade) -> None:
        """Handle trade from active feed."""
        for callback in self._trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")

    def _handle_bar(self, bar: Bar) -> None:
        """Handle bar from active feed."""
        for callback in self._bar_callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Error in bar callback: {e}")

    def _handle_error(self, feed_name: str, error: Exception) -> None:
        """Handle error from a feed."""
        logger.error(f"Error from feed '{feed_name}': {error}")

        info = self._feeds.get(feed_name)
        if info:
            info['last_error'] = error
            info['error_count'] += 1

        # Forward to callbacks
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

        # Trigger failover if this is the active feed
        if feed_name == self._active_feed_name and self.failover_enabled:
            asyncio.create_task(self._failover())

    def _handle_status_change(
        self,
        feed_name: str,
        status: DataFeedStatus
    ) -> None:
        """Handle status change from a feed."""
        logger.info(f"Feed '{feed_name}' status changed to {status.value}")

        info = self._feeds.get(feed_name)
        if info:
            info['healthy'] = status == DataFeedStatus.CONNECTED

        # Trigger failover if active feed disconnected
        if (feed_name == self._active_feed_name and
            status in (DataFeedStatus.ERROR, DataFeedStatus.DISCONNECTED) and
            self.failover_enabled):
            asyncio.create_task(self._failover())

    async def _health_monitor(self) -> None:
        """Monitor feed health and trigger failover if needed."""
        try:
            while self._running:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Check active feed health
                if self._active_feed_name:
                    info = self._feeds[self._active_feed_name]
                    feed = info['feed']

                    if not feed.is_connected:
                        logger.warning(f"Active feed '{self._active_feed_name}' is not connected")
                        info['healthy'] = False

                        if self.failover_enabled:
                            await self._failover()

        except asyncio.CancelledError:
            logger.debug("Health monitor cancelled")
            raise
        except Exception as e:
            logger.error(f"Health monitor error: {e}")
