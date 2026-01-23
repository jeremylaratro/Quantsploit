"""
Broker Integration for Quantsploit Live Trading

This module provides a comprehensive broker integration layer with:
- Abstract BrokerInterface for multiple broker support
- AlpacaBroker implementation with full API integration
- OrderManager for order lifecycle management
- RiskManager for pre-trade risk checks
- PositionReconciler for state synchronization

Safety Features:
- Paper trading by default
- Explicit confirmation required for live mode
- Maximum order size limits
- Daily loss circuit breaker
- Rate limiting on API calls
- Retry logic for transient failures

Author: Quantsploit Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
import threading
import time
import logging
import hashlib
import json
import os
from collections import deque


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class OrderSide(Enum):
    """Order side (buy or sell)"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class TimeInForce(Enum):
    """Time in force for orders"""
    DAY = "day"  # Valid until end of day
    GTC = "gtc"  # Good til cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    OPG = "opg"  # At open
    CLS = "cls"  # At close


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"  # Not yet submitted
    SUBMITTED = "submitted"  # Submitted to broker
    ACCEPTED = "accepted"  # Accepted by broker
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"  # Failed to submit


class TradingMode(Enum):
    """Trading mode"""
    PAPER = "paper"  # Paper trading (default, safe)
    LIVE = "live"  # Live trading with real money


class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Order:
    """Represents a trading order"""
    id: str  # Internal order ID
    broker_order_id: Optional[str] = None  # Broker's order ID
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: int = 0
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_avg_price: float = 0.0
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    reject_reason: Optional[str] = None
    client_order_id: Optional[str] = None  # For idempotency
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary"""
        return {
            "id": self.id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_avg_price": self.filled_avg_price,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: int  # Positive for long, negative for short
    avg_entry_price: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    side: PositionSide = PositionSide.FLAT
    cost_basis: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def update_price(self, price: float):
        """Update position with current market price"""
        self.current_price = price
        self.market_value = self.quantity * price
        if self.cost_basis != 0:
            self.unrealized_pnl = self.market_value - self.cost_basis
            self.unrealized_pnl_pct = (self.unrealized_pnl / abs(self.cost_basis)) * 100
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "side": self.side.value,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "cost_basis": self.cost_basis,
        }


@dataclass
class Account:
    """Represents broker account information"""
    account_id: str = ""
    buying_power: float = 0.0
    cash: float = 0.0
    portfolio_value: float = 0.0
    equity: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    day_trade_count: int = 0
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    account_blocked: bool = False
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert account to dictionary"""
        return {
            "account_id": self.account_id,
            "buying_power": self.buying_power,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
            "equity": self.equity,
            "margin_used": self.margin_used,
            "margin_available": self.margin_available,
            "day_trade_count": self.day_trade_count,
            "pattern_day_trader": self.pattern_day_trader,
            "trading_blocked": self.trading_blocked,
        }


@dataclass
class RiskLimits:
    """Risk management limits"""
    # Position limits
    max_position_size: int = 1000  # Max shares per position
    max_position_value: float = 50000.0  # Max dollar value per position
    max_total_positions: int = 10  # Max number of concurrent positions
    max_position_pct_portfolio: float = 0.20  # Max 20% of portfolio per position

    # Order limits
    max_order_size: int = 500  # Max shares per order
    max_order_value: float = 25000.0  # Max dollar value per order
    max_daily_orders: int = 100  # Max orders per day

    # Loss limits
    max_daily_loss: float = 5000.0  # Max daily loss before circuit breaker
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss
    max_position_loss_pct: float = 0.10  # Max 10% loss per position before forced exit
    max_drawdown_pct: float = 0.15  # Max 15% drawdown from peak

    # Trading restrictions
    allowed_symbols: Optional[List[str]] = None  # If set, only these symbols can be traded
    blocked_symbols: List[str] = field(default_factory=list)  # Symbols that cannot be traded
    min_price: float = 1.0  # Minimum stock price
    max_price: float = 10000.0  # Maximum stock price

    # Time restrictions
    allow_premarket: bool = False
    allow_afterhours: bool = False


@dataclass
class BrokerConfig:
    """Broker configuration"""
    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    mode: TradingMode = TradingMode.PAPER
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_calls: int = 200  # Max calls per minute
    risk_limits: RiskLimits = field(default_factory=RiskLimits)


# =============================================================================
# Abstract Broker Interface
# =============================================================================

class BrokerInterface(ABC):
    """
    Abstract base class for broker integrations.

    All broker implementations must inherit from this class and implement
    the abstract methods. This provides a consistent interface for:
    - Connection management
    - Order management (place, cancel, modify)
    - Position and account queries
    - Order status monitoring

    Safety features are built into the interface:
    - Paper trading by default
    - Confirmation required for live mode
    - Rate limiting
    - Error handling
    """

    def __init__(self, config: BrokerConfig):
        self.config = config
        self._connected = False
        self._connection_lock = threading.Lock()
        self._order_callbacks: List[Callable[[Order], None]] = []

        # Safety: Default to paper trading
        if config.mode == TradingMode.LIVE:
            logger.warning("LIVE TRADING MODE - Real money at risk!")
        else:
            logger.info("Paper trading mode - No real money at risk")

    @property
    def is_connected(self) -> bool:
        """Check if connected to broker"""
        return self._connected

    @property
    def trading_mode(self) -> TradingMode:
        """Get current trading mode"""
        return self.config.mode

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to broker.

        Returns:
            bool: True if connection successful

        Raises:
            BrokerConnectionError: If connection fails
        """
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from broker.

        Returns:
            bool: True if disconnection successful
        """
        pass

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """
        Submit an order to the broker.

        Args:
            order: Order object with order details

        Returns:
            Order: Updated order with broker order ID and status

        Raises:
            OrderValidationError: If order fails validation
            BrokerError: If broker rejects order
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Internal order ID or broker order ID

        Returns:
            bool: True if cancellation successful

        Raises:
            OrderNotFoundError: If order not found
            BrokerError: If cancellation fails
        """
        pass

    @abstractmethod
    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify
            quantity: New quantity (optional)
            limit_price: New limit price (optional)
            stop_price: New stop price (optional)

        Returns:
            Order: Updated order

        Raises:
            OrderNotFoundError: If order not found
            OrderModificationError: If modification fails
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """
        Get current status of an order.

        Args:
            order_id: Order ID to query

        Returns:
            Order: Order with current status

        Raises:
            OrderNotFoundError: If order not found
        """
        pass

    @abstractmethod
    def get_all_orders(
        self,
        status: Optional[OrderStatus] = None,
        since: Optional[datetime] = None
    ) -> List[Order]:
        """
        Get all orders, optionally filtered.

        Args:
            status: Filter by order status
            since: Only orders since this time

        Returns:
            List[Order]: List of orders
        """
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.

        Returns:
            Dict[str, Position]: Positions keyed by symbol
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position or None if no position
        """
        pass

    @abstractmethod
    def get_account(self) -> Account:
        """
        Get account information.

        Returns:
            Account: Current account state
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Dict[str, float]:
        """
        Get current quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with 'bid', 'ask', 'last' prices
        """
        pass

    def register_order_callback(self, callback: Callable[[Order], None]):
        """Register callback for order updates"""
        self._order_callbacks.append(callback)

    def _notify_order_update(self, order: Order):
        """Notify all registered callbacks of order update"""
        for callback in self._order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Order callback error: {e}")


# =============================================================================
# Custom Exceptions
# =============================================================================

class BrokerError(Exception):
    """Base exception for broker errors"""
    pass


class BrokerConnectionError(BrokerError):
    """Connection to broker failed"""
    pass


class OrderValidationError(BrokerError):
    """Order failed validation"""
    pass


class OrderNotFoundError(BrokerError):
    """Order not found"""
    pass


class OrderModificationError(BrokerError):
    """Order modification failed"""
    pass


class RiskLimitExceededError(BrokerError):
    """Risk limit exceeded"""
    pass


class InsufficientFundsError(BrokerError):
    """Insufficient funds for order"""
    pass


class LiveTradingNotConfirmedError(BrokerError):
    """Live trading requires explicit confirmation"""
    pass


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Rate limiter using token bucket algorithm.

    Prevents exceeding broker API rate limits.
    """

    def __init__(self, calls_per_minute: int = 200):
        self.calls_per_minute = calls_per_minute
        self.calls = deque()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire permission to make an API call.

        Args:
            timeout: Max time to wait for permission

        Returns:
            bool: True if permission granted
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            with self._lock:
                now = time.time()

                # Remove calls older than 1 minute
                while self.calls and self.calls[0] < now - 60:
                    self.calls.popleft()

                # Check if under limit
                if len(self.calls) < self.calls_per_minute:
                    self.calls.append(now)
                    return True

            # Wait a bit before retrying
            time.sleep(0.1)

        return False

    def get_remaining(self) -> int:
        """Get remaining calls available in current window"""
        with self._lock:
            now = time.time()
            while self.calls and self.calls[0] < now - 60:
                self.calls.popleft()
            return self.calls_per_minute - len(self.calls)


# =============================================================================
# Alpaca Broker Implementation
# =============================================================================

class AlpacaBroker(BrokerInterface):
    """
    Alpaca Markets broker integration.

    Provides full integration with Alpaca's trading API:
    - Paper and live trading support
    - Market, limit, stop, stop-limit orders
    - Position and account management
    - Real-time order status updates

    Safety Features:
    - Paper trading by default
    - Confirmation dialog for live mode
    - Rate limiting
    - Retry logic for transient failures

    Usage:
        config = BrokerConfig(
            api_key=os.environ['ALPACA_API_KEY'],
            api_secret=os.environ['ALPACA_API_SECRET'],
            mode=TradingMode.PAPER  # Default safe mode
        )
        broker = AlpacaBroker(config)
        broker.connect()

        # Place a market order
        order = Order(
            id="order_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10
        )
        result = broker.place_order(order)
    """

    # Alpaca API endpoints
    PAPER_BASE_URL = "https://paper-api.alpaca.markets"
    LIVE_BASE_URL = "https://api.alpaca.markets"

    def __init__(
        self,
        config: BrokerConfig,
        live_trading_confirmed: bool = False
    ):
        """
        Initialize Alpaca broker.

        Args:
            config: Broker configuration
            live_trading_confirmed: Must be True to enable live trading
        """
        # Set base URL based on mode
        if config.mode == TradingMode.LIVE:
            if not live_trading_confirmed:
                raise LiveTradingNotConfirmedError(
                    "Live trading requires explicit confirmation. "
                    "Set live_trading_confirmed=True to acknowledge real money risk."
                )
            config.base_url = self.LIVE_BASE_URL
            logger.warning("=" * 60)
            logger.warning("LIVE TRADING MODE ENABLED - REAL MONEY AT RISK")
            logger.warning("=" * 60)
        else:
            config.base_url = self.PAPER_BASE_URL
            logger.info("Paper trading mode enabled (no real money)")

        super().__init__(config)

        self._api = None
        self._rate_limiter = RateLimiter(config.rate_limit_calls)
        self._orders: Dict[str, Order] = {}  # Internal order tracking
        self._live_confirmed = live_trading_confirmed

    def connect(self) -> bool:
        """Connect to Alpaca API"""
        with self._connection_lock:
            if self._connected:
                logger.info("Already connected to Alpaca")
                return True

            try:
                # Import alpaca-trade-api
                try:
                    import alpaca_trade_api as tradeapi
                except ImportError:
                    raise BrokerConnectionError(
                        "alpaca-trade-api package not installed. "
                        "Install with: pip install alpaca-trade-api"
                    )

                # Create API connection
                self._api = tradeapi.REST(
                    key_id=self.config.api_key,
                    secret_key=self.config.api_secret,
                    base_url=self.config.base_url,
                    api_version='v2'
                )

                # Test connection by getting account
                account = self._api.get_account()
                logger.info(f"Connected to Alpaca account: {account.id}")
                logger.info(f"Account status: {account.status}")
                logger.info(f"Buying power: ${float(account.buying_power):,.2f}")

                self._connected = True
                return True

            except Exception as e:
                logger.error(f"Failed to connect to Alpaca: {e}")
                raise BrokerConnectionError(f"Connection failed: {e}")

    def disconnect(self) -> bool:
        """Disconnect from Alpaca API"""
        with self._connection_lock:
            self._api = None
            self._connected = False
            logger.info("Disconnected from Alpaca")
            return True

    def _ensure_connected(self):
        """Ensure broker is connected"""
        if not self._connected or self._api is None:
            raise BrokerConnectionError("Not connected to broker")

    def _rate_limit(self):
        """Apply rate limiting"""
        if not self._rate_limiter.acquire(timeout=30):
            raise BrokerError("Rate limit exceeded, try again later")

    def _retry_with_backoff(
        self,
        func: Callable,
        *args,
        max_retries: int = None,
        **kwargs
    ):
        """
        Execute function with exponential backoff retry.

        Args:
            func: Function to execute
            max_retries: Override default max retries

        Returns:
            Result of function
        """
        max_retries = max_retries or self.config.max_retries
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")

        raise last_error

    def place_order(self, order: Order) -> Order:
        """Submit order to Alpaca"""
        self._ensure_connected()

        try:
            # Map order type to Alpaca format
            alpaca_order_type = order.order_type.value

            # Prepare order parameters
            order_params = {
                "symbol": order.symbol,
                "qty": order.quantity,
                "side": order.side.value,
                "type": alpaca_order_type,
                "time_in_force": order.time_in_force.value,
            }

            # Add price parameters for limit/stop orders
            if order.order_type == OrderType.LIMIT:
                if order.limit_price is None:
                    raise OrderValidationError("Limit price required for limit order")
                order_params["limit_price"] = str(order.limit_price)

            elif order.order_type == OrderType.STOP:
                if order.stop_price is None:
                    raise OrderValidationError("Stop price required for stop order")
                order_params["stop_price"] = str(order.stop_price)

            elif order.order_type == OrderType.STOP_LIMIT:
                if order.limit_price is None or order.stop_price is None:
                    raise OrderValidationError(
                        "Both limit and stop prices required for stop-limit order"
                    )
                order_params["limit_price"] = str(order.limit_price)
                order_params["stop_price"] = str(order.stop_price)

            # Add client order ID for idempotency
            if order.client_order_id:
                order_params["client_order_id"] = order.client_order_id

            # Submit order with retry
            def submit():
                return self._api.submit_order(**order_params)

            alpaca_order = self._retry_with_backoff(submit)

            # Update order with broker response
            order.broker_order_id = alpaca_order.id
            order.status = self._map_alpaca_status(alpaca_order.status)
            order.submitted_at = datetime.now()
            order.last_updated = datetime.now()

            if alpaca_order.filled_qty:
                order.filled_quantity = int(alpaca_order.filled_qty)
            if alpaca_order.filled_avg_price:
                order.filled_avg_price = float(alpaca_order.filled_avg_price)

            # Track order internally
            self._orders[order.id] = order

            logger.info(
                f"Order submitted: {order.side.value} {order.quantity} "
                f"{order.symbol} @ {order.order_type.value} "
                f"(ID: {order.broker_order_id})"
            )

            # Notify callbacks
            self._notify_order_update(order)

            return order

        except Exception as e:
            order.status = OrderStatus.FAILED
            order.reject_reason = str(e)
            order.last_updated = datetime.now()
            logger.error(f"Order failed: {e}")
            raise BrokerError(f"Order submission failed: {e}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        self._ensure_connected()

        try:
            # Find the broker order ID
            broker_order_id = order_id
            if order_id in self._orders:
                broker_order_id = self._orders[order_id].broker_order_id

            def cancel():
                self._api.cancel_order(broker_order_id)

            self._retry_with_backoff(cancel)

            # Update internal tracking
            if order_id in self._orders:
                self._orders[order_id].status = OrderStatus.CANCELLED
                self._orders[order_id].cancelled_at = datetime.now()
                self._notify_order_update(self._orders[order_id])

            logger.info(f"Order cancelled: {broker_order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            raise BrokerError(f"Order cancellation failed: {e}")

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """
        Modify an existing order.

        Alpaca supports order replacement for modifying orders.
        """
        self._ensure_connected()

        try:
            # Get current order
            current_order = self.get_order_status(order_id)

            if current_order.status not in [
                OrderStatus.PENDING,
                OrderStatus.SUBMITTED,
                OrderStatus.ACCEPTED,
                OrderStatus.PARTIALLY_FILLED
            ]:
                raise OrderModificationError(
                    f"Cannot modify order in status: {current_order.status.value}"
                )

            # Prepare replacement parameters
            replace_params = {}
            if quantity is not None:
                replace_params["qty"] = str(quantity)
            if limit_price is not None:
                replace_params["limit_price"] = str(limit_price)
            if stop_price is not None:
                replace_params["stop_price"] = str(stop_price)

            if not replace_params:
                raise OrderModificationError("No modifications specified")

            # Get broker order ID
            broker_order_id = current_order.broker_order_id or order_id

            def replace():
                return self._api.replace_order(broker_order_id, **replace_params)

            new_alpaca_order = self._retry_with_backoff(replace)

            # Update order
            current_order.broker_order_id = new_alpaca_order.id
            if quantity is not None:
                current_order.quantity = quantity
            if limit_price is not None:
                current_order.limit_price = limit_price
            if stop_price is not None:
                current_order.stop_price = stop_price
            current_order.last_updated = datetime.now()

            logger.info(f"Order modified: {broker_order_id}")
            self._notify_order_update(current_order)

            return current_order

        except Exception as e:
            logger.error(f"Failed to modify order: {e}")
            raise OrderModificationError(f"Order modification failed: {e}")

    def get_order_status(self, order_id: str) -> Order:
        """Get current status of an order"""
        self._ensure_connected()

        try:
            # Check internal tracking first
            if order_id in self._orders:
                order = self._orders[order_id]
                broker_order_id = order.broker_order_id
            else:
                broker_order_id = order_id
                order = Order(id=order_id)

            def get_order():
                return self._api.get_order(broker_order_id)

            alpaca_order = self._retry_with_backoff(get_order)

            # Update order from Alpaca response
            order.broker_order_id = alpaca_order.id
            order.symbol = alpaca_order.symbol
            order.side = OrderSide(alpaca_order.side)
            order.order_type = OrderType(alpaca_order.type)
            order.quantity = int(alpaca_order.qty)
            order.status = self._map_alpaca_status(alpaca_order.status)

            if alpaca_order.limit_price:
                order.limit_price = float(alpaca_order.limit_price)
            if alpaca_order.stop_price:
                order.stop_price = float(alpaca_order.stop_price)
            if alpaca_order.filled_qty:
                order.filled_quantity = int(alpaca_order.filled_qty)
            if alpaca_order.filled_avg_price:
                order.filled_avg_price = float(alpaca_order.filled_avg_price)
            if alpaca_order.filled_at:
                order.filled_at = alpaca_order.filled_at

            order.last_updated = datetime.now()

            # Update internal tracking
            if order.id in self._orders:
                self._orders[order.id] = order

            return order

        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            raise OrderNotFoundError(f"Order not found: {order_id}")

    def get_all_orders(
        self,
        status: Optional[OrderStatus] = None,
        since: Optional[datetime] = None
    ) -> List[Order]:
        """Get all orders with optional filtering"""
        self._ensure_connected()

        try:
            # Build filter parameters
            params = {}
            if status:
                params["status"] = status.value
            if since:
                params["after"] = since.isoformat()

            def list_orders():
                return self._api.list_orders(**params)

            alpaca_orders = self._retry_with_backoff(list_orders)

            orders = []
            for ao in alpaca_orders:
                order = Order(
                    id=ao.id,
                    broker_order_id=ao.id,
                    symbol=ao.symbol,
                    side=OrderSide(ao.side),
                    order_type=OrderType(ao.type),
                    quantity=int(ao.qty),
                    status=self._map_alpaca_status(ao.status),
                )

                if ao.limit_price:
                    order.limit_price = float(ao.limit_price)
                if ao.stop_price:
                    order.stop_price = float(ao.stop_price)
                if ao.filled_qty:
                    order.filled_quantity = int(ao.filled_qty)
                if ao.filled_avg_price:
                    order.filled_avg_price = float(ao.filled_avg_price)

                orders.append(order)

            return orders

        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            raise BrokerError(f"Failed to get orders: {e}")

    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        self._ensure_connected()

        try:
            def list_positions():
                return self._api.list_positions()

            alpaca_positions = self._retry_with_backoff(list_positions)

            positions = {}
            for ap in alpaca_positions:
                qty = int(ap.qty)
                position = Position(
                    symbol=ap.symbol,
                    quantity=qty,
                    avg_entry_price=float(ap.avg_entry_price),
                    current_price=float(ap.current_price),
                    market_value=float(ap.market_value),
                    unrealized_pnl=float(ap.unrealized_pl),
                    unrealized_pnl_pct=float(ap.unrealized_plpc) * 100,
                    cost_basis=float(ap.cost_basis),
                    side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
                )
                positions[ap.symbol] = position

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            raise BrokerError(f"Failed to get positions: {e}")

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        self._ensure_connected()

        try:
            def get_pos():
                return self._api.get_position(symbol)

            ap = self._retry_with_backoff(get_pos)

            qty = int(ap.qty)
            return Position(
                symbol=ap.symbol,
                quantity=qty,
                avg_entry_price=float(ap.avg_entry_price),
                current_price=float(ap.current_price),
                market_value=float(ap.market_value),
                unrealized_pnl=float(ap.unrealized_pl),
                unrealized_pnl_pct=float(ap.unrealized_plpc) * 100,
                cost_basis=float(ap.cost_basis),
                side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
            )

        except Exception as e:
            if "position does not exist" in str(e).lower():
                return None
            logger.error(f"Failed to get position: {e}")
            raise BrokerError(f"Failed to get position: {e}")

    def get_account(self) -> Account:
        """Get account information"""
        self._ensure_connected()

        try:
            def get_acct():
                return self._api.get_account()

            aa = self._retry_with_backoff(get_acct)

            return Account(
                account_id=aa.id,
                buying_power=float(aa.buying_power),
                cash=float(aa.cash),
                portfolio_value=float(aa.portfolio_value),
                equity=float(aa.equity),
                margin_used=float(aa.initial_margin) if aa.initial_margin else 0,
                margin_available=float(aa.buying_power),
                day_trade_count=int(aa.daytrade_count),
                pattern_day_trader=aa.pattern_day_trader,
                trading_blocked=aa.trading_blocked,
                account_blocked=aa.account_blocked,
            )

        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            raise BrokerError(f"Failed to get account: {e}")

    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get current quote for a symbol"""
        self._ensure_connected()

        try:
            def get_latest():
                return self._api.get_latest_quote(symbol)

            quote = self._retry_with_backoff(get_latest)

            return {
                "bid": float(quote.bp) if quote.bp else 0.0,
                "ask": float(quote.ap) if quote.ap else 0.0,
                "bid_size": int(quote.bs) if quote.bs else 0,
                "ask_size": int(quote.as_) if quote.as_ else 0,
            }

        except Exception as e:
            logger.error(f"Failed to get quote: {e}")
            raise BrokerError(f"Failed to get quote: {e}")

    def _map_alpaca_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca order status to internal status"""
        status_map = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.ACCEPTED,
            "pending_new": OrderStatus.PENDING,
            "accepted_for_bidding": OrderStatus.ACCEPTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
            "pending_cancel": OrderStatus.CANCELLED,
            "pending_replace": OrderStatus.ACCEPTED,
            "stopped": OrderStatus.FILLED,
            "suspended": OrderStatus.REJECTED,
            "calculated": OrderStatus.ACCEPTED,
        }
        return status_map.get(alpaca_status.lower(), OrderStatus.PENDING)

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders (emergency function).

        Returns:
            int: Number of orders cancelled
        """
        self._ensure_connected()

        try:
            def cancel_all():
                return self._api.cancel_all_orders()

            result = self._retry_with_backoff(cancel_all)

            # Update internal tracking
            for order in self._orders.values():
                if order.status in [
                    OrderStatus.PENDING,
                    OrderStatus.SUBMITTED,
                    OrderStatus.ACCEPTED,
                    OrderStatus.PARTIALLY_FILLED
                ]:
                    order.status = OrderStatus.CANCELLED
                    order.cancelled_at = datetime.now()

            cancelled_count = len(result) if result else 0
            logger.warning(f"EMERGENCY: Cancelled {cancelled_count} orders")
            return cancelled_count

        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            raise BrokerError(f"Failed to cancel all orders: {e}")

    def close_all_positions(self) -> int:
        """
        Close all positions (emergency function).

        Returns:
            int: Number of positions closed
        """
        self._ensure_connected()

        try:
            def close_all():
                return self._api.close_all_positions()

            result = self._retry_with_backoff(close_all)

            closed_count = len(result) if result else 0
            logger.warning(f"EMERGENCY: Closed {closed_count} positions")
            return closed_count

        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            raise BrokerError(f"Failed to close all positions: {e}")


# =============================================================================
# Order Manager
# =============================================================================

class OrderManager:
    """
    Manages order lifecycle and tracking.

    Features:
    - Track all orders (pending, filled, cancelled)
    - Order validation before submission
    - Rate limiting
    - Retry logic for transient failures
    - Idempotency support via client order IDs

    Usage:
        manager = OrderManager(broker)

        # Submit order with validation
        order = manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=150.00
        )
        result = manager.submit_order(order)

        # Get order status
        status = manager.get_order(order.id)

        # Get all active orders
        active = manager.get_active_orders()
    """

    def __init__(
        self,
        broker: BrokerInterface,
        max_orders_per_day: int = 100
    ):
        self.broker = broker
        self.max_orders_per_day = max_orders_per_day

        # Order tracking
        self._orders: Dict[str, Order] = {}
        self._orders_by_broker_id: Dict[str, str] = {}  # broker_id -> internal_id
        self._daily_order_count = 0
        self._daily_reset_date = date.today()

        # Threading
        self._lock = threading.RLock()
        self._order_counter = 0

        # Register for order updates from broker
        broker.register_order_callback(self._on_order_update)

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        with self._lock:
            self._order_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            return f"ORD_{timestamp}_{self._order_counter:06d}"

    def _generate_client_order_id(self, order: Order) -> str:
        """Generate idempotent client order ID"""
        # Hash of order details for idempotency
        data = f"{order.symbol}_{order.side.value}_{order.quantity}_{order.order_type.value}_{datetime.now().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:24]

    def _check_daily_reset(self):
        """Reset daily counters if new day"""
        today = date.today()
        if today != self._daily_reset_date:
            self._daily_order_count = 0
            self._daily_reset_date = today

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        metadata: Dict[str, Any] = None
    ) -> Order:
        """
        Create a new order (does not submit).

        Args:
            symbol: Stock symbol
            side: Buy or sell
            quantity: Number of shares
            order_type: Type of order
            limit_price: Limit price (for limit/stop-limit orders)
            stop_price: Stop price (for stop/stop-limit orders)
            time_in_force: Order duration
            metadata: Additional order metadata

        Returns:
            Order object ready for submission
        """
        order_id = self._generate_order_id()

        order = Order(
            id=order_id,
            symbol=symbol.upper(),
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            status=OrderStatus.PENDING,
            metadata=metadata or {}
        )

        # Generate client order ID for idempotency
        order.client_order_id = self._generate_client_order_id(order)

        return order

    def validate_order(self, order: Order) -> Tuple[bool, str]:
        """
        Validate order before submission.

        Args:
            order: Order to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []

        # Basic validation
        if not order.symbol:
            errors.append("Symbol is required")

        if order.quantity <= 0:
            errors.append("Quantity must be positive")

        # Order type specific validation
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None or order.limit_price <= 0:
                errors.append("Limit price required for limit order")

        if order.order_type == OrderType.STOP:
            if order.stop_price is None or order.stop_price <= 0:
                errors.append("Stop price required for stop order")

        if order.order_type == OrderType.STOP_LIMIT:
            if order.limit_price is None or order.limit_price <= 0:
                errors.append("Limit price required for stop-limit order")
            if order.stop_price is None or order.stop_price <= 0:
                errors.append("Stop price required for stop-limit order")

        # Check daily order limit
        self._check_daily_reset()
        if self._daily_order_count >= self.max_orders_per_day:
            errors.append(f"Daily order limit reached ({self.max_orders_per_day})")

        if errors:
            return False, "; ".join(errors)

        return True, ""

    def submit_order(self, order: Order) -> Order:
        """
        Submit order to broker.

        Args:
            order: Order to submit

        Returns:
            Updated order with broker response

        Raises:
            OrderValidationError: If validation fails
            BrokerError: If submission fails
        """
        with self._lock:
            # Validate order
            is_valid, error_msg = self.validate_order(order)
            if not is_valid:
                order.status = OrderStatus.REJECTED
                order.reject_reason = error_msg
                raise OrderValidationError(error_msg)

            # Store order before submission
            self._orders[order.id] = order

            try:
                # Submit to broker
                result = self.broker.place_order(order)

                # Update tracking
                self._daily_order_count += 1
                if result.broker_order_id:
                    self._orders_by_broker_id[result.broker_order_id] = order.id

                logger.info(
                    f"Order submitted: {order.id} -> {result.broker_order_id} "
                    f"({result.status.value})"
                )

                return result

            except Exception as e:
                order.status = OrderStatus.FAILED
                order.reject_reason = str(e)
                raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        with self._lock:
            order = self._orders.get(order_id)
            if not order:
                raise OrderNotFoundError(f"Order not found: {order_id}")

            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                logger.warning(f"Cannot cancel order in status: {order.status.value}")
                return False

            return self.broker.cancel_order(order_id)

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        with self._lock:
            return self._orders.get(order_id)

    def get_order_by_broker_id(self, broker_order_id: str) -> Optional[Order]:
        """Get order by broker order ID"""
        with self._lock:
            internal_id = self._orders_by_broker_id.get(broker_order_id)
            if internal_id:
                return self._orders.get(internal_id)
            return None

    def get_active_orders(self) -> List[Order]:
        """Get all active (non-terminal) orders"""
        with self._lock:
            return [
                order for order in self._orders.values()
                if order.status in [
                    OrderStatus.PENDING,
                    OrderStatus.SUBMITTED,
                    OrderStatus.ACCEPTED,
                    OrderStatus.PARTIALLY_FILLED
                ]
            ]

    def get_filled_orders(self, since: Optional[datetime] = None) -> List[Order]:
        """Get filled orders, optionally since a time"""
        with self._lock:
            filled = [
                order for order in self._orders.values()
                if order.status == OrderStatus.FILLED
            ]
            if since:
                filled = [o for o in filled if o.filled_at and o.filled_at >= since]
            return filled

    def get_all_orders(self) -> List[Order]:
        """Get all tracked orders"""
        with self._lock:
            return list(self._orders.values())

    def _on_order_update(self, order: Order):
        """Handle order update from broker"""
        with self._lock:
            if order.id in self._orders:
                self._orders[order.id] = order
            elif order.broker_order_id in self._orders_by_broker_id:
                internal_id = self._orders_by_broker_id[order.broker_order_id]
                self._orders[internal_id] = order

    def sync_with_broker(self) -> int:
        """
        Synchronize orders with broker.

        Returns:
            int: Number of orders synchronized
        """
        try:
            broker_orders = self.broker.get_all_orders()

            with self._lock:
                for bo in broker_orders:
                    if bo.broker_order_id not in self._orders_by_broker_id:
                        # New order from broker
                        self._orders[bo.id] = bo
                        self._orders_by_broker_id[bo.broker_order_id] = bo.id
                    else:
                        # Update existing order
                        internal_id = self._orders_by_broker_id[bo.broker_order_id]
                        self._orders[internal_id] = bo

            return len(broker_orders)

        except Exception as e:
            logger.error(f"Failed to sync orders: {e}")
            return 0


# =============================================================================
# Risk Manager
# =============================================================================

class RiskManager:
    """
    Pre-trade risk management and monitoring.

    Features:
    - Pre-trade risk checks before order submission
    - Position limits (per symbol and total)
    - Daily loss limits with circuit breaker
    - Emergency stop (cancel all orders, close all positions)
    - Real-time P&L monitoring

    Usage:
        risk_mgr = RiskManager(broker, limits=RiskLimits(
            max_position_value=25000,
            max_daily_loss=2500,
            max_daily_loss_pct=0.05
        ))

        # Check if order passes risk limits
        approved, reason = risk_mgr.check_order(order, current_price)
        if not approved:
            print(f"Order rejected: {reason}")

        # Monitor P&L
        if risk_mgr.check_circuit_breaker():
            risk_mgr.emergency_stop()
    """

    def __init__(
        self,
        broker: BrokerInterface,
        limits: RiskLimits = None
    ):
        self.broker = broker
        self.limits = limits or RiskLimits()

        # Daily tracking
        self._daily_pnl = 0.0
        self._daily_orders = 0
        self._daily_reset_date = date.today()
        self._peak_equity = 0.0

        # State
        self._circuit_breaker_triggered = False
        self._lock = threading.RLock()

        # Callbacks for risk events
        self._risk_callbacks: List[Callable[[str, Dict], None]] = []

    def register_risk_callback(self, callback: Callable[[str, Dict], None]):
        """Register callback for risk events"""
        self._risk_callbacks.append(callback)

    def _notify_risk_event(self, event_type: str, details: Dict):
        """Notify callbacks of risk event"""
        for callback in self._risk_callbacks:
            try:
                callback(event_type, details)
            except Exception as e:
                logger.error(f"Risk callback error: {e}")

    def _check_daily_reset(self):
        """Reset daily tracking if new day"""
        today = date.today()
        if today != self._daily_reset_date:
            self._daily_pnl = 0.0
            self._daily_orders = 0
            self._daily_reset_date = today
            self._circuit_breaker_triggered = False
            logger.info("Daily risk limits reset")

    def check_order(
        self,
        order: Order,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Check if order passes risk limits.

        Args:
            order: Order to check
            current_price: Current market price

        Returns:
            Tuple of (approved, reason)
        """
        with self._lock:
            self._check_daily_reset()

            reasons = []

            # Check circuit breaker
            if self._circuit_breaker_triggered:
                return False, "Circuit breaker triggered - trading halted"

            # Check symbol restrictions
            if self.limits.blocked_symbols and order.symbol in self.limits.blocked_symbols:
                return False, f"Symbol {order.symbol} is blocked"

            if self.limits.allowed_symbols and order.symbol not in self.limits.allowed_symbols:
                return False, f"Symbol {order.symbol} not in allowed list"

            # Check price limits
            if current_price < self.limits.min_price:
                return False, f"Price ${current_price:.2f} below minimum ${self.limits.min_price:.2f}"

            if current_price > self.limits.max_price:
                return False, f"Price ${current_price:.2f} above maximum ${self.limits.max_price:.2f}"

            # Check order size limits
            if order.quantity > self.limits.max_order_size:
                return False, f"Order size {order.quantity} exceeds max {self.limits.max_order_size}"

            order_value = order.quantity * current_price
            if order_value > self.limits.max_order_value:
                return False, f"Order value ${order_value:,.2f} exceeds max ${self.limits.max_order_value:,.2f}"

            # Check daily order limit
            if self._daily_orders >= self.limits.max_daily_orders:
                return False, f"Daily order limit reached ({self.limits.max_daily_orders})"

            # Check position limits
            try:
                positions = self.broker.get_positions()
                account = self.broker.get_account()

                # Check total position count
                if len(positions) >= self.limits.max_total_positions:
                    # Only reject if this is a new position
                    if order.side == OrderSide.BUY and order.symbol not in positions:
                        return False, f"Max positions ({self.limits.max_total_positions}) reached"

                # Check position size for this symbol
                existing_qty = 0
                if order.symbol in positions:
                    existing_qty = positions[order.symbol].quantity

                if order.side == OrderSide.BUY:
                    new_qty = existing_qty + order.quantity
                else:
                    new_qty = existing_qty - order.quantity

                if abs(new_qty) > self.limits.max_position_size:
                    return False, f"Position size {abs(new_qty)} exceeds max {self.limits.max_position_size}"

                new_position_value = abs(new_qty) * current_price
                if new_position_value > self.limits.max_position_value:
                    return False, f"Position value ${new_position_value:,.2f} exceeds max ${self.limits.max_position_value:,.2f}"

                # Check position as percentage of portfolio
                if account.portfolio_value > 0:
                    position_pct = new_position_value / account.portfolio_value
                    if position_pct > self.limits.max_position_pct_portfolio:
                        return False, (
                            f"Position {position_pct:.1%} exceeds max "
                            f"{self.limits.max_position_pct_portfolio:.1%} of portfolio"
                        )

                # Check if we have buying power
                if order.side == OrderSide.BUY:
                    if order_value > account.buying_power:
                        return False, f"Insufficient buying power (${account.buying_power:,.2f})"

            except Exception as e:
                logger.warning(f"Could not verify position limits: {e}")

            # All checks passed
            self._daily_orders += 1
            return True, "Order approved"

    def check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker should be triggered.

        Returns:
            bool: True if circuit breaker is triggered
        """
        with self._lock:
            self._check_daily_reset()

            if self._circuit_breaker_triggered:
                return True

            try:
                account = self.broker.get_account()
                positions = self.broker.get_positions()

                # Calculate daily P&L
                total_unrealized_pnl = sum(
                    p.unrealized_pnl for p in positions.values()
                )
                current_pnl = total_unrealized_pnl  # + realized P&L if tracked

                # Check absolute daily loss
                if current_pnl < -self.limits.max_daily_loss:
                    self._trigger_circuit_breaker(
                        f"Daily loss ${abs(current_pnl):,.2f} exceeds limit "
                        f"${self.limits.max_daily_loss:,.2f}"
                    )
                    return True

                # Check percentage daily loss
                if account.portfolio_value > 0:
                    loss_pct = abs(current_pnl) / account.portfolio_value
                    if current_pnl < 0 and loss_pct > self.limits.max_daily_loss_pct:
                        self._trigger_circuit_breaker(
                            f"Daily loss {loss_pct:.1%} exceeds limit "
                            f"{self.limits.max_daily_loss_pct:.1%}"
                        )
                        return True

                # Check drawdown from peak
                if account.equity > self._peak_equity:
                    self._peak_equity = account.equity

                if self._peak_equity > 0:
                    drawdown = (self._peak_equity - account.equity) / self._peak_equity
                    if drawdown > self.limits.max_drawdown_pct:
                        self._trigger_circuit_breaker(
                            f"Drawdown {drawdown:.1%} exceeds limit "
                            f"{self.limits.max_drawdown_pct:.1%}"
                        )
                        return True

                # Check individual position losses
                for symbol, position in positions.items():
                    if position.cost_basis > 0:
                        loss_pct = abs(position.unrealized_pnl) / position.cost_basis
                        if position.unrealized_pnl < 0 and loss_pct > self.limits.max_position_loss_pct:
                            logger.warning(
                                f"Position {symbol} loss {loss_pct:.1%} exceeds limit - "
                                f"consider closing"
                            )

            except Exception as e:
                logger.error(f"Error checking circuit breaker: {e}")

            return False

    def _trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker"""
        self._circuit_breaker_triggered = True
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")

        self._notify_risk_event("circuit_breaker", {
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })

    def emergency_stop(self) -> Dict[str, int]:
        """
        Emergency stop - cancel all orders and close all positions.

        Returns:
            Dict with counts of cancelled orders and closed positions
        """
        logger.critical("EMERGENCY STOP INITIATED")

        results = {
            "orders_cancelled": 0,
            "positions_closed": 0,
            "errors": []
        }

        # Cancel all orders
        try:
            if isinstance(self.broker, AlpacaBroker):
                results["orders_cancelled"] = self.broker.cancel_all_orders()
            else:
                orders = self.broker.get_all_orders(status=OrderStatus.ACCEPTED)
                for order in orders:
                    try:
                        self.broker.cancel_order(order.id)
                        results["orders_cancelled"] += 1
                    except Exception as e:
                        results["errors"].append(f"Cancel order {order.id}: {e}")
        except Exception as e:
            results["errors"].append(f"Cancel all orders: {e}")

        # Close all positions
        try:
            if isinstance(self.broker, AlpacaBroker):
                results["positions_closed"] = self.broker.close_all_positions()
            else:
                positions = self.broker.get_positions()
                for symbol, position in positions.items():
                    try:
                        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                        close_order = Order(
                            id=f"EMERGENCY_CLOSE_{symbol}",
                            symbol=symbol,
                            side=side,
                            order_type=OrderType.MARKET,
                            quantity=abs(position.quantity)
                        )
                        self.broker.place_order(close_order)
                        results["positions_closed"] += 1
                    except Exception as e:
                        results["errors"].append(f"Close position {symbol}: {e}")
        except Exception as e:
            results["errors"].append(f"Close all positions: {e}")

        self._notify_risk_event("emergency_stop", results)

        logger.critical(
            f"Emergency stop complete: {results['orders_cancelled']} orders cancelled, "
            f"{results['positions_closed']} positions closed"
        )

        return results

    def reset_circuit_breaker(self, confirm: bool = False) -> bool:
        """
        Reset circuit breaker (requires confirmation).

        Args:
            confirm: Must be True to reset

        Returns:
            bool: True if reset successful
        """
        if not confirm:
            logger.warning("Circuit breaker reset requires confirm=True")
            return False

        with self._lock:
            self._circuit_breaker_triggered = False
            logger.warning("Circuit breaker has been reset")
            return True

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        with self._lock:
            self._check_daily_reset()

            try:
                account = self.broker.get_account()
                positions = self.broker.get_positions()

                total_unrealized = sum(p.unrealized_pnl for p in positions.values())
                total_position_value = sum(p.market_value for p in positions.values())

                return {
                    "circuit_breaker_triggered": self._circuit_breaker_triggered,
                    "daily_orders": self._daily_orders,
                    "max_daily_orders": self.limits.max_daily_orders,
                    "daily_pnl": total_unrealized,
                    "max_daily_loss": self.limits.max_daily_loss,
                    "portfolio_value": account.portfolio_value,
                    "buying_power": account.buying_power,
                    "position_count": len(positions),
                    "max_positions": self.limits.max_total_positions,
                    "total_position_value": total_position_value,
                    "peak_equity": self._peak_equity,
                    "current_equity": account.equity,
                    "drawdown": (
                        (self._peak_equity - account.equity) / self._peak_equity
                        if self._peak_equity > 0 else 0
                    )
                }
            except Exception as e:
                logger.error(f"Error getting risk status: {e}")
                return {
                    "circuit_breaker_triggered": self._circuit_breaker_triggered,
                    "error": str(e)
                }


# =============================================================================
# Position Reconciler
# =============================================================================

class PositionReconciler:
    """
    Reconciles internal position state with broker state.

    Features:
    - Compare internal state vs broker state
    - Detect and handle discrepancies
    - Sync positions on startup
    - Periodic reconciliation

    Usage:
        reconciler = PositionReconciler(broker, order_manager)

        # Sync on startup
        discrepancies = reconciler.sync()

        # Periodic check
        discrepancies = reconciler.check_discrepancies()
        for d in discrepancies:
            print(f"Discrepancy: {d}")
    """

    def __init__(
        self,
        broker: BrokerInterface,
        order_manager: Optional[OrderManager] = None,
        tolerance: float = 0.01  # 1% tolerance for price differences
    ):
        self.broker = broker
        self.order_manager = order_manager
        self.tolerance = tolerance

        # Internal position tracking
        self._internal_positions: Dict[str, Position] = {}
        self._lock = threading.RLock()

        # Reconciliation history
        self._last_reconcile: Optional[datetime] = None
        self._discrepancy_history: List[Dict] = []

    def sync(self) -> List[Dict]:
        """
        Synchronize internal state with broker.

        Returns:
            List of discrepancies found and resolved
        """
        with self._lock:
            discrepancies = []

            try:
                broker_positions = self.broker.get_positions()

                # Check for positions in broker but not internal
                for symbol, broker_pos in broker_positions.items():
                    if symbol not in self._internal_positions:
                        discrepancies.append({
                            "type": "missing_internal",
                            "symbol": symbol,
                            "broker_qty": broker_pos.quantity,
                            "internal_qty": 0,
                            "action": "added_to_internal"
                        })
                        self._internal_positions[symbol] = broker_pos
                    else:
                        # Check quantity mismatch
                        internal_pos = self._internal_positions[symbol]
                        if internal_pos.quantity != broker_pos.quantity:
                            discrepancies.append({
                                "type": "quantity_mismatch",
                                "symbol": symbol,
                                "broker_qty": broker_pos.quantity,
                                "internal_qty": internal_pos.quantity,
                                "action": "updated_internal"
                            })
                            self._internal_positions[symbol] = broker_pos

                # Check for positions in internal but not broker
                for symbol in list(self._internal_positions.keys()):
                    if symbol not in broker_positions:
                        discrepancies.append({
                            "type": "missing_broker",
                            "symbol": symbol,
                            "broker_qty": 0,
                            "internal_qty": self._internal_positions[symbol].quantity,
                            "action": "removed_from_internal"
                        })
                        del self._internal_positions[symbol]

                self._last_reconcile = datetime.now()
                self._discrepancy_history.extend(discrepancies)

                if discrepancies:
                    logger.warning(f"Position reconciliation found {len(discrepancies)} discrepancies")
                    for d in discrepancies:
                        logger.warning(f"  {d['type']}: {d['symbol']} - {d['action']}")
                else:
                    logger.info("Position reconciliation complete - no discrepancies")

                return discrepancies

            except Exception as e:
                logger.error(f"Position reconciliation failed: {e}")
                return [{
                    "type": "error",
                    "error": str(e)
                }]

    def check_discrepancies(self) -> List[Dict]:
        """
        Check for discrepancies without auto-resolving.

        Returns:
            List of discrepancies found
        """
        with self._lock:
            discrepancies = []

            try:
                broker_positions = self.broker.get_positions()

                # Check for differences
                all_symbols = set(self._internal_positions.keys()) | set(broker_positions.keys())

                for symbol in all_symbols:
                    internal_qty = self._internal_positions.get(symbol, Position(symbol, 0, 0)).quantity
                    broker_qty = broker_positions.get(symbol, Position(symbol, 0, 0)).quantity

                    if internal_qty != broker_qty:
                        discrepancies.append({
                            "symbol": symbol,
                            "internal_qty": internal_qty,
                            "broker_qty": broker_qty,
                            "difference": broker_qty - internal_qty,
                            "timestamp": datetime.now().isoformat()
                        })

                return discrepancies

            except Exception as e:
                logger.error(f"Discrepancy check failed: {e}")
                return [{
                    "type": "error",
                    "error": str(e)
                }]

    def update_position(
        self,
        symbol: str,
        quantity_change: int,
        price: float,
        side: OrderSide
    ):
        """
        Update internal position tracking after order fill.

        Args:
            symbol: Stock symbol
            quantity_change: Number of shares (positive)
            price: Fill price
            side: Buy or sell
        """
        with self._lock:
            if symbol not in self._internal_positions:
                self._internal_positions[symbol] = Position(
                    symbol=symbol,
                    quantity=0,
                    avg_entry_price=0.0
                )

            pos = self._internal_positions[symbol]

            if side == OrderSide.BUY:
                # Update average entry price
                if pos.quantity >= 0:
                    total_cost = pos.avg_entry_price * pos.quantity + price * quantity_change
                    new_quantity = pos.quantity + quantity_change
                    pos.avg_entry_price = total_cost / new_quantity if new_quantity > 0 else 0
                pos.quantity += quantity_change
            else:
                pos.quantity -= quantity_change

            pos.cost_basis = pos.avg_entry_price * pos.quantity

            # Determine side
            if pos.quantity > 0:
                pos.side = PositionSide.LONG
            elif pos.quantity < 0:
                pos.side = PositionSide.SHORT
            else:
                pos.side = PositionSide.FLAT
                del self._internal_positions[symbol]  # Remove flat positions

    def get_internal_positions(self) -> Dict[str, Position]:
        """Get internal position tracking"""
        with self._lock:
            return dict(self._internal_positions)

    def get_reconciliation_history(self, limit: int = 100) -> List[Dict]:
        """Get recent reconciliation history"""
        with self._lock:
            return self._discrepancy_history[-limit:]


# =============================================================================
# Factory Functions
# =============================================================================

def create_paper_broker(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    risk_limits: Optional[RiskLimits] = None
) -> AlpacaBroker:
    """
    Create an Alpaca paper trading broker.

    Args:
        api_key: Alpaca API key (or from ALPACA_API_KEY env var)
        api_secret: Alpaca API secret (or from ALPACA_SECRET_KEY env var)
        risk_limits: Optional risk limits

    Returns:
        AlpacaBroker configured for paper trading
    """
    config = BrokerConfig(
        api_key=api_key or os.environ.get('ALPACA_API_KEY', ''),
        api_secret=api_secret or os.environ.get('ALPACA_SECRET_KEY', ''),
        mode=TradingMode.PAPER,
        risk_limits=risk_limits or RiskLimits()
    )

    return AlpacaBroker(config)


def create_live_broker(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    risk_limits: Optional[RiskLimits] = None,
    confirm_live_trading: bool = False
) -> AlpacaBroker:
    """
    Create an Alpaca live trading broker.

    WARNING: This uses real money!

    Args:
        api_key: Alpaca API key (or from ALPACA_API_KEY env var)
        api_secret: Alpaca API secret (or from ALPACA_SECRET_KEY env var)
        risk_limits: Optional risk limits (recommended for safety)
        confirm_live_trading: Must be True to acknowledge real money risk

    Returns:
        AlpacaBroker configured for live trading
    """
    if not confirm_live_trading:
        raise LiveTradingNotConfirmedError(
            "Live trading requires confirm_live_trading=True to acknowledge risk"
        )

    config = BrokerConfig(
        api_key=api_key or os.environ.get('ALPACA_API_KEY', ''),
        api_secret=api_secret or os.environ.get('ALPACA_SECRET_KEY', ''),
        mode=TradingMode.LIVE,
        risk_limits=risk_limits or RiskLimits()
    )

    return AlpacaBroker(config, live_trading_confirmed=True)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    'OrderSide',
    'OrderType',
    'TimeInForce',
    'OrderStatus',
    'TradingMode',
    'PositionSide',

    # Data classes
    'Order',
    'Position',
    'Account',
    'RiskLimits',
    'BrokerConfig',

    # Interfaces and implementations
    'BrokerInterface',
    'AlpacaBroker',
    'OrderManager',
    'RiskManager',
    'PositionReconciler',
    'RateLimiter',

    # Exceptions
    'BrokerError',
    'BrokerConnectionError',
    'OrderValidationError',
    'OrderNotFoundError',
    'OrderModificationError',
    'RiskLimitExceededError',
    'InsufficientFundsError',
    'LiveTradingNotConfirmedError',

    # Factory functions
    'create_paper_broker',
    'create_live_broker',
]
