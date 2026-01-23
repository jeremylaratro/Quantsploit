"""
Structured Logging System for Quantsploit

This module provides a comprehensive structured logging system with:
- JSON-formatted structured logging
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Contextual fields (strategy, symbol, timestamp)
- File and console handlers
- Log rotation (daily and size-based)
- Performance tracking and metrics aggregation
- Specialized log methods for trading operations
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import time
import statistics


class LogLevel(Enum):
    """Log levels supported by QuantsploitLogger"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class EventType(Enum):
    """Event types for structured logging"""
    TRADE = "trade"
    SIGNAL = "signal"
    BACKTEST_START = "backtest_start"
    BACKTEST_END = "backtest_end"
    ERROR = "error"
    PERFORMANCE = "performance"
    POSITION = "position"
    SYSTEM = "system"
    CONFIG = "config"
    DATA = "data"


@dataclass
class LogConfig:
    """Configuration for the logging system"""
    log_dir: str = "logs"
    log_level: LogLevel = LogLevel.INFO
    console_enabled: bool = True
    file_enabled: bool = True
    json_format: bool = True

    # File rotation settings
    rotation_type: str = "time"  # "time" or "size"
    max_bytes: int = 10 * 1024 * 1024  # 10 MB for size-based rotation
    backup_count: int = 30  # Number of backup files to keep
    rotation_when: str = "midnight"  # For time-based rotation
    rotation_interval: int = 1  # Days for time-based rotation

    # Performance tracking
    track_performance: bool = True
    metrics_flush_interval: int = 60  # Seconds between metrics aggregation

    # Environment-based settings
    environment: str = "development"  # "development", "production", "testing"

    def __post_init__(self):
        """Apply environment-specific defaults"""
        if self.environment == "production":
            self.log_level = LogLevel.INFO
            self.json_format = True
            self.file_enabled = True
        elif self.environment == "development":
            self.log_level = LogLevel.DEBUG
            self.json_format = True
            self.console_enabled = True
        elif self.environment == "testing":
            self.log_level = LogLevel.DEBUG
            self.console_enabled = False
            self.file_enabled = False


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging

    Produces log entries in the format:
    {
        "timestamp": "2026-01-22T10:30:00Z",
        "level": "INFO",
        "event_type": "trade",
        "strategy": "kalman_adaptive",
        "symbol": "AAPL",
        "message": "Long entry signal triggered",
        ...additional_fields
    }
    """

    def __init__(self, include_traceback: bool = True):
        super().__init__()
        self.include_traceback = include_traceback

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add custom fields from record
        if hasattr(record, "event_type"):
            log_entry["event_type"] = record.event_type

        if hasattr(record, "context"):
            log_entry.update(record.context)

        # Add exception info if present
        if record.exc_info and self.include_traceback:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add source location
        log_entry["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName
        }

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Colored console formatter for human-readable output
    """

    COLORS = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console with colors"""
        color = self.COLORS.get(record.levelno, self.RESET)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        base_msg = f"{color}[{timestamp}] [{record.levelname}] {record.getMessage()}{self.RESET}"

        # Add context info if available
        if hasattr(record, "context") and record.context:
            context_str = " | ".join(f"{k}={v}" for k, v in record.context.items() if k not in ["message"])
            if context_str:
                base_msg += f" | {context_str}"

        # Add traceback if present
        if record.exc_info:
            base_msg += f"\n{self.RESET}{''.join(traceback.format_exception(*record.exc_info))}"

        return base_msg


@dataclass
class MetricSample:
    """Single metric sample"""
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time window"""
    count: int = 0
    total: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    samples: List[float] = field(default_factory=list)

    def add_sample(self, value: float):
        """Add a sample to the aggregation"""
        self.count += 1
        self.total += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.samples.append(value)

    @property
    def mean(self) -> float:
        """Calculate mean of samples"""
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def std_dev(self) -> float:
        """Calculate standard deviation of samples"""
        if len(self.samples) < 2:
            return 0.0
        return statistics.stdev(self.samples)

    @property
    def p50(self) -> float:
        """Calculate 50th percentile (median)"""
        if not self.samples:
            return 0.0
        return statistics.median(self.samples)

    @property
    def p95(self) -> float:
        """Calculate 95th percentile"""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(0.95 * len(sorted_samples))
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def p99(self) -> float:
        """Calculate 99th percentile"""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(0.99 * len(sorted_samples))
        return sorted_samples[min(idx, len(sorted_samples) - 1)]


class PerformanceTracker:
    """
    Performance tracking and metrics aggregation

    Tracks:
    - Latency metrics (signal generation, trade execution)
    - Error rates
    - Trade counts and outcomes
    - Custom metrics
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: Dict[str, AggregatedMetrics] = defaultdict(AggregatedMetrics)
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._start_time = datetime.now(timezone.utc)

    def record_latency(self, operation: str, latency_ms: float, labels: Optional[Dict[str, str]] = None):
        """Record latency for an operation"""
        metric_name = f"latency_{operation}"
        if labels:
            label_str = "_".join(f"{k}_{v}" for k, v in sorted(labels.items()))
            metric_name = f"{metric_name}_{label_str}"

        with self._lock:
            self._metrics[metric_name].add_sample(latency_ms)

    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a counter"""
        with self._lock:
            self._counters[counter_name] += value

    def set_gauge(self, gauge_name: str, value: float):
        """Set a gauge value"""
        with self._lock:
            self._gauges[gauge_name] = value

    def get_counter(self, counter_name: str) -> int:
        """Get current counter value"""
        with self._lock:
            return self._counters[counter_name]

    def get_gauge(self, gauge_name: str) -> Optional[float]:
        """Get current gauge value"""
        with self._lock:
            return self._gauges.get(gauge_name)

    def record_trade(self, outcome: str, strategy: str, symbol: str):
        """Record a trade outcome"""
        with self._lock:
            self._counters[f"trades_total"] += 1
            self._counters[f"trades_{outcome}"] += 1
            self._counters[f"trades_{strategy}_{outcome}"] += 1
            self._counters[f"trades_{symbol}_{outcome}"] += 1

    def record_error(self, error_type: str, strategy: Optional[str] = None):
        """Record an error occurrence"""
        with self._lock:
            self._counters["errors_total"] += 1
            self._counters[f"errors_{error_type}"] += 1
            if strategy:
                self._counters[f"errors_{strategy}_{error_type}"] += 1

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics"""
        with self._lock:
            summary = {
                "uptime_seconds": (datetime.now(timezone.utc) - self._start_time).total_seconds(),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "latencies": {}
            }

            for metric_name, agg in self._metrics.items():
                if agg.count > 0:
                    summary["latencies"][metric_name] = {
                        "count": agg.count,
                        "mean_ms": round(agg.mean, 3),
                        "min_ms": round(agg.min_value, 3),
                        "max_ms": round(agg.max_value, 3),
                        "std_dev_ms": round(agg.std_dev, 3),
                        "p50_ms": round(agg.p50, 3),
                        "p95_ms": round(agg.p95, 3),
                        "p99_ms": round(agg.p99, 3),
                    }

            return summary

    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._start_time = datetime.now(timezone.utc)

    def context_timer(self, operation: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        return _TimerContext(self, operation, labels)


class _TimerContext:
    """Context manager for timing operations"""

    def __init__(self, tracker: PerformanceTracker, operation: str, labels: Optional[Dict[str, str]] = None):
        self.tracker = tracker
        self.operation = operation
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.tracker.record_latency(self.operation, elapsed_ms, self.labels)
        return False


class QuantsploitLogger:
    """
    Main structured logging class for Quantsploit

    Features:
    - JSON-formatted structured logging
    - Multiple log levels
    - Contextual fields (strategy, symbol, timestamp)
    - File and console handlers
    - Specialized methods for trading operations
    - Performance tracking integration
    """

    _instance: Optional['QuantsploitLogger'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for global logger access"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, name: str = "quantsploit", config: Optional[LogConfig] = None):
        """
        Initialize the logger

        Args:
            name: Logger name
            config: Logging configuration
        """
        # Prevent re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.name = name
        self.config = config or LogConfig()
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self.config.log_level.value)
        self._logger.handlers.clear()

        self._performance_tracker = PerformanceTracker()
        self._context: Dict[str, Any] = {}

        self._setup_handlers()
        self._initialized = True

    def _setup_handlers(self):
        """Set up logging handlers based on configuration"""
        # Console handler
        if self.config.console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.config.log_level.value)

            if self.config.json_format:
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(ConsoleFormatter())

            self._logger.addHandler(console_handler)

        # File handler
        if self.config.file_enabled:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / f"{self.name}.log"

            if self.config.rotation_type == "size":
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=self.config.max_bytes,
                    backupCount=self.config.backup_count
                )
            else:  # time-based rotation
                file_handler = logging.handlers.TimedRotatingFileHandler(
                    log_file,
                    when=self.config.rotation_when,
                    interval=self.config.rotation_interval,
                    backupCount=self.config.backup_count
                )

            file_handler.setLevel(self.config.log_level.value)
            file_handler.setFormatter(JSONFormatter())
            self._logger.addHandler(file_handler)

    def set_context(self, **kwargs):
        """Set persistent context fields that will be included in all log entries"""
        self._context.update(kwargs)

    def clear_context(self, *keys):
        """Clear specific context fields or all context if no keys provided"""
        if keys:
            for key in keys:
                self._context.pop(key, None)
        else:
            self._context.clear()

    def _log(self, level: int, message: str, event_type: Optional[EventType] = None, **kwargs):
        """Internal logging method"""
        # Merge context with kwargs
        context = {**self._context, **kwargs}

        # Create log record with extra fields
        extra = {
            "event_type": event_type.value if event_type else EventType.SYSTEM.value,
            "context": context
        }

        self._logger.log(level, message, extra=extra)

    # Standard log level methods
    def debug(self, message: str, **kwargs):
        """Log at DEBUG level"""
        self._log(logging.DEBUG, message, EventType.SYSTEM, **kwargs)

    def info(self, message: str, **kwargs):
        """Log at INFO level"""
        self._log(logging.INFO, message, EventType.SYSTEM, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log at WARNING level"""
        self._log(logging.WARNING, message, EventType.SYSTEM, **kwargs)

    def error(self, message: str, **kwargs):
        """Log at ERROR level"""
        self._log(logging.ERROR, message, EventType.ERROR, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log at CRITICAL level"""
        self._log(logging.CRITICAL, message, EventType.ERROR, **kwargs)

    # Specialized trading log methods
    def log_trade(
        self,
        action: str,
        symbol: str,
        shares: int,
        price: float,
        strategy: Optional[str] = None,
        order_type: str = "market",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        commission: Optional[float] = None,
        slippage: Optional[float] = None,
        position_value: Optional[float] = None,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        message: Optional[str] = None,
        **kwargs
    ):
        """
        Log a trade entry or exit with full details

        Args:
            action: Trade action (BUY, SELL, SHORT, COVER)
            symbol: Trading symbol
            shares: Number of shares
            price: Execution price
            strategy: Strategy name
            order_type: Order type (market, limit, stop)
            stop_loss: Stop loss price if set
            take_profit: Take profit price if set
            commission: Commission paid
            slippage: Slippage incurred
            position_value: Total position value
            pnl: Profit/loss (for exits)
            pnl_pct: Profit/loss percentage (for exits)
            message: Additional message
            **kwargs: Additional fields
        """
        trade_data = {
            "action": action.upper(),
            "symbol": symbol,
            "shares": shares,
            "price": price,
            "order_type": order_type,
        }

        if strategy:
            trade_data["strategy"] = strategy
        if stop_loss is not None:
            trade_data["stop_loss"] = stop_loss
        if take_profit is not None:
            trade_data["take_profit"] = take_profit
        if commission is not None:
            trade_data["commission"] = commission
        if slippage is not None:
            trade_data["slippage"] = slippage
        if position_value is not None:
            trade_data["position_value"] = position_value
        if pnl is not None:
            trade_data["pnl"] = pnl
        if pnl_pct is not None:
            trade_data["pnl_pct"] = pnl_pct

        trade_data.update(kwargs)

        log_message = message or f"{action.upper()} {shares} shares of {symbol} at ${price:.2f}"

        self._log(logging.INFO, log_message, EventType.TRADE, **trade_data)

        # Track trade in performance metrics
        if self.config.track_performance:
            outcome = "entry" if action.upper() in ["BUY", "SHORT"] else "exit"
            self._performance_tracker.record_trade(outcome, strategy or "unknown", symbol)

    def log_signal(
        self,
        signal_type: str,
        symbol: str,
        strategy: str,
        strength: Optional[float] = None,
        confidence: Optional[float] = None,
        indicators: Optional[Dict[str, float]] = None,
        conditions: Optional[List[str]] = None,
        message: Optional[str] = None,
        **kwargs
    ):
        """
        Log strategy signal generation

        Args:
            signal_type: Signal type (BUY, SELL, HOLD, etc.)
            symbol: Trading symbol
            strategy: Strategy name
            strength: Signal strength (0-1)
            confidence: Signal confidence (0-1)
            indicators: Indicator values that triggered signal
            conditions: List of conditions that were met
            message: Additional message
            **kwargs: Additional fields
        """
        signal_data = {
            "signal_type": signal_type.upper(),
            "symbol": symbol,
            "strategy": strategy,
        }

        if strength is not None:
            signal_data["strength"] = strength
        if confidence is not None:
            signal_data["confidence"] = confidence
        if indicators:
            signal_data["indicators"] = indicators
        if conditions:
            signal_data["conditions"] = conditions

        signal_data.update(kwargs)

        log_message = message or f"{signal_type.upper()} signal for {symbol} from {strategy}"

        self._log(logging.INFO, log_message, EventType.SIGNAL, **signal_data)

        # Track signal generation
        if self.config.track_performance:
            self._performance_tracker.increment_counter(f"signals_{signal_type.lower()}")
            self._performance_tracker.increment_counter(f"signals_{strategy}")

    def log_backtest_start(
        self,
        strategy: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        initial_capital: float,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Log backtest start

        Args:
            strategy: Strategy name
            symbols: List of symbols being tested
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            parameters: Strategy parameters
            **kwargs: Additional fields
        """
        backtest_data = {
            "strategy": strategy,
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
        }

        if parameters:
            backtest_data["parameters"] = parameters

        backtest_data.update(kwargs)

        log_message = f"Starting backtest for {strategy} on {len(symbols)} symbols from {start_date} to {end_date}"

        self._log(logging.INFO, log_message, EventType.BACKTEST_START, **backtest_data)

        # Track backtest start
        if self.config.track_performance:
            self._performance_tracker.increment_counter("backtests_started")

    def log_backtest_end(
        self,
        strategy: str,
        total_return: float,
        total_return_pct: float,
        sharpe_ratio: float,
        max_drawdown: float,
        total_trades: int,
        win_rate: float,
        duration_seconds: Optional[float] = None,
        additional_metrics: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Log backtest completion

        Args:
            strategy: Strategy name
            total_return: Total dollar return
            total_return_pct: Total return percentage
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown percentage
            total_trades: Number of trades executed
            win_rate: Win rate percentage
            duration_seconds: Backtest execution time
            additional_metrics: Additional performance metrics
            **kwargs: Additional fields
        """
        backtest_data = {
            "strategy": strategy,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "win_rate": win_rate,
        }

        if duration_seconds is not None:
            backtest_data["duration_seconds"] = duration_seconds
        if additional_metrics:
            backtest_data["additional_metrics"] = additional_metrics

        backtest_data.update(kwargs)

        log_message = f"Backtest completed for {strategy}: {total_return_pct:.2f}% return, {sharpe_ratio:.3f} Sharpe, {total_trades} trades"

        self._log(logging.INFO, log_message, EventType.BACKTEST_END, **backtest_data)

        # Track backtest completion
        if self.config.track_performance:
            self._performance_tracker.increment_counter("backtests_completed")
            if duration_seconds:
                self._performance_tracker.record_latency("backtest_execution", duration_seconds * 1000)

    def log_error(
        self,
        error: Exception,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Log an error with full traceback

        Args:
            error: Exception object
            strategy: Strategy name if applicable
            symbol: Symbol if applicable
            operation: Operation that failed
            context: Additional context about the error
            **kwargs: Additional fields
        """
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }

        if strategy:
            error_data["strategy"] = strategy
        if symbol:
            error_data["symbol"] = symbol
        if operation:
            error_data["operation"] = operation
        if context:
            error_data["error_context"] = context

        error_data.update(kwargs)

        log_message = f"Error in {operation or 'operation'}: {type(error).__name__}: {str(error)}"

        self._log(logging.ERROR, log_message, EventType.ERROR, **error_data)

        # Track error
        if self.config.track_performance:
            self._performance_tracker.record_error(type(error).__name__, strategy)

    def log_performance(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
        period: Optional[str] = None,
        **kwargs
    ):
        """
        Log periodic performance metrics

        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            strategy: Strategy name if applicable
            symbol: Symbol if applicable
            period: Time period (e.g., "1h", "1d", "1w")
            **kwargs: Additional fields
        """
        perf_data = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
        }

        if strategy:
            perf_data["strategy"] = strategy
        if symbol:
            perf_data["symbol"] = symbol
        if period:
            perf_data["period"] = period

        perf_data.update(kwargs)

        log_message = f"Performance metric: {metric_name} = {value}{unit}"

        self._log(logging.INFO, log_message, EventType.PERFORMANCE, **perf_data)

    def log_position(
        self,
        action: str,
        symbol: str,
        shares: int,
        entry_price: float,
        current_price: float,
        unrealized_pnl: float,
        strategy: Optional[str] = None,
        **kwargs
    ):
        """
        Log position update

        Args:
            action: Position action (OPEN, UPDATE, CLOSE)
            symbol: Trading symbol
            shares: Number of shares
            entry_price: Entry price
            current_price: Current price
            unrealized_pnl: Unrealized P&L
            strategy: Strategy name
            **kwargs: Additional fields
        """
        position_data = {
            "action": action.upper(),
            "symbol": symbol,
            "shares": shares,
            "entry_price": entry_price,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
        }

        if strategy:
            position_data["strategy"] = strategy

        position_data.update(kwargs)

        log_message = f"Position {action.upper()}: {shares} {symbol} @ ${entry_price:.2f}, unrealized P&L: ${unrealized_pnl:.2f}"

        self._log(logging.INFO, log_message, EventType.POSITION, **position_data)

    def log_data_fetch(
        self,
        source: str,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        records_fetched: Optional[int] = None,
        duration_ms: Optional[float] = None,
        **kwargs
    ):
        """
        Log data fetch operations

        Args:
            source: Data source (e.g., "yfinance", "alpaca")
            symbols: List of symbols fetched
            start_date: Start date of data
            end_date: End date of data
            records_fetched: Number of records fetched
            duration_ms: Fetch duration in milliseconds
            **kwargs: Additional fields
        """
        data_info = {
            "source": source,
            "symbols": symbols,
            "symbol_count": len(symbols),
        }

        if start_date:
            data_info["start_date"] = start_date
        if end_date:
            data_info["end_date"] = end_date
        if records_fetched is not None:
            data_info["records_fetched"] = records_fetched
        if duration_ms is not None:
            data_info["duration_ms"] = duration_ms

        data_info.update(kwargs)

        log_message = f"Data fetched from {source}: {len(symbols)} symbols, {records_fetched or 'N/A'} records"

        self._log(logging.INFO, log_message, EventType.DATA, **data_info)

        # Track fetch latency
        if self.config.track_performance and duration_ms:
            self._performance_tracker.record_latency("data_fetch", duration_ms, {"source": source})

    @property
    def performance_tracker(self) -> PerformanceTracker:
        """Get the performance tracker instance"""
        return self._performance_tracker

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        return self._performance_tracker.get_metrics_summary()

    def log_performance_summary(self):
        """Log current performance summary"""
        summary = self.get_performance_summary()
        self._log(logging.INFO, "Performance summary", EventType.PERFORMANCE, **summary)


# Global logger instance
_global_logger: Optional[QuantsploitLogger] = None


def setup_logging(
    name: str = "quantsploit",
    log_dir: str = "logs",
    log_level: str = "INFO",
    console_enabled: bool = True,
    file_enabled: bool = True,
    json_format: bool = True,
    rotation_type: str = "time",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 30,
    environment: str = "development"
) -> QuantsploitLogger:
    """
    Initialize logging with configuration

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_enabled: Enable console output
        file_enabled: Enable file output
        json_format: Use JSON format for console output
        rotation_type: "time" for daily rotation, "size" for size-based
        max_bytes: Max file size for size-based rotation
        backup_count: Number of backup files to keep
        environment: Environment name (development, production, testing)

    Returns:
        Configured QuantsploitLogger instance
    """
    global _global_logger

    # Parse log level
    level_map = {
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARNING": LogLevel.WARNING,
        "ERROR": LogLevel.ERROR,
        "CRITICAL": LogLevel.CRITICAL,
    }
    level = level_map.get(log_level.upper(), LogLevel.INFO)

    config = LogConfig(
        log_dir=log_dir,
        log_level=level,
        console_enabled=console_enabled,
        file_enabled=file_enabled,
        json_format=json_format,
        rotation_type=rotation_type,
        max_bytes=max_bytes,
        backup_count=backup_count,
        environment=environment
    )

    # Reset singleton for reconfiguration
    QuantsploitLogger._instance = None

    _global_logger = QuantsploitLogger(name=name, config=config)
    return _global_logger


def get_logger() -> QuantsploitLogger:
    """
    Get the global logger instance

    Returns:
        QuantsploitLogger instance (creates default if not initialized)
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = setup_logging()

    return _global_logger


def log_function_call(
    include_args: bool = True,
    include_result: bool = False,
    log_level: str = "DEBUG"
) -> Callable:
    """
    Decorator to log function calls with timing

    Args:
        include_args: Include function arguments in log
        include_result: Include function result in log
        log_level: Log level to use

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            logger = get_logger()
            start_time = time.perf_counter()

            log_data = {
                "function": func.__name__,
                "module": func.__module__,
            }

            if include_args:
                # Safely convert args to strings
                log_data["args"] = [str(a)[:100] for a in args]
                log_data["kwargs"] = {k: str(v)[:100] for k, v in kwargs.items()}

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                log_data["duration_ms"] = elapsed_ms
                log_data["status"] = "success"

                if include_result:
                    log_data["result"] = str(result)[:200]

                getattr(logger, log_level.lower())(
                    f"Function {func.__name__} completed in {elapsed_ms:.2f}ms",
                    **log_data
                )

                # Track latency
                if logger.config.track_performance:
                    logger.performance_tracker.record_latency(
                        f"function_{func.__name__}",
                        elapsed_ms
                    )

                return result

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                log_data["duration_ms"] = elapsed_ms
                log_data["status"] = "error"

                logger.log_error(e, operation=func.__name__, context=log_data)
                raise

        return wrapper
    return decorator


# Convenience exports
__all__ = [
    'QuantsploitLogger',
    'LogConfig',
    'LogLevel',
    'EventType',
    'PerformanceTracker',
    'JSONFormatter',
    'ConsoleFormatter',
    'setup_logging',
    'get_logger',
    'log_function_call',
]
