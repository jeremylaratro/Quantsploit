"""
Advanced Backtesting Framework for Quantsploit

This module provides a comprehensive backtesting engine with:
- Position management
- Commission and slippage modeling
- Advanced risk metrics (Sharpe, Sortino, Calmar, Max Drawdown, etc.)
- Trade statistics and analytics
- Portfolio management
- Walk-forward optimization support
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class OrderType(Enum):
    """Order types supported by the backtesting engine"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Represents a single trade"""
    entry_date: datetime
    exit_date: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: int = 0
    side: PositionSide = PositionSide.LONG
    commission: float = 0.0
    slippage: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion

    def close(self, exit_date: datetime, exit_price: float, commission: float, slippage: float):
        """Close the trade and calculate P&L"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.commission += commission
        self.slippage += slippage

        if self.side == PositionSide.LONG:
            self.pnl = (self.exit_price - self.entry_price) * self.shares - self.commission - self.slippage
            self.pnl_pct = ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            self.pnl = (self.entry_price - self.exit_price) * self.shares - self.commission - self.slippage
            self.pnl_pct = ((self.entry_price - self.exit_price) / self.entry_price) * 100


@dataclass
class Position:
    """Represents a current position"""
    symbol: str
    side: PositionSide
    shares: int
    entry_price: float
    entry_date: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion

    def update(self, current_price: float):
        """Update position with current price"""
        self.current_price = current_price

        if self.side == PositionSide.LONG:
            pnl = (current_price - self.entry_price) * self.shares
            self.unrealized_pnl = pnl

            # Update MAE/MFE
            if pnl < self.mae:
                self.mae = pnl
            if pnl > self.mfe:
                self.mfe = pnl
        else:  # SHORT
            pnl = (self.entry_price - current_price) * self.shares
            self.unrealized_pnl = pnl

            # Update MAE/MFE
            if pnl < self.mae:
                self.mae = pnl
            if pnl > self.mfe:
                self.mfe = pnl


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 1000.0
    commission_pct: float = 0.001  # 0.1% per trade
    commission_min: float = 1.0  # Minimum $1 commission
    slippage_pct: float = 0.001  # 0.1% slippage
    position_size: float = 1.0  # Fraction of capital per trade (1.0 = 100%)
    max_positions: int = 1  # Maximum concurrent positions
    margin_requirement: float = 1.0  # 1.0 = no margin, 0.5 = 2x leverage
    risk_free_rate: float = 0.02  # Annual risk-free rate for Sharpe calculation
    benchmark_symbol: str = "SPY"  # Benchmark for comparison


@dataclass
class BacktestResults:
    """Results from a backtest run"""
    # Performance metrics
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    benchmark_return: float = 0.0
    alpha: float = 0.0  # Excess return vs benchmark
    beta: float = 0.0  # Correlation to benchmark

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0
    downside_deviation: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0
    avg_mae: float = 0.0  # Average Maximum Adverse Excursion
    avg_mfe: float = 0.0  # Average Maximum Favorable Excursion

    # Equity curve
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[Trade] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert results to dictionary for display"""
        return {
            "Total Return": f"${self.total_return:,.2f} ({self.total_return_pct:.2f}%)",
            "Annualized Return": f"{self.annualized_return:.2f}%",
            "Benchmark Return": f"{self.benchmark_return:.2f}%",
            "Alpha": f"{self.alpha:.2f}%",
            "Beta": f"{self.beta:.3f}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Sortino Ratio": f"{self.sortino_ratio:.3f}",
            "Calmar Ratio": f"{self.calmar_ratio:.3f}",
            "Max Drawdown": f"{self.max_drawdown:.2f}%",
            "Max DD Duration": f"{self.max_drawdown_duration} days",
            "Volatility": f"{self.volatility:.2f}%",
            "Total Trades": self.total_trades,
            "Win Rate": f"{self.win_rate:.2f}%",
            "Profit Factor": f"{self.profit_factor:.3f}",
            "Expectancy": f"${self.expectancy:.2f}",
            "Avg Win": f"${self.avg_win:.2f} ({self.avg_win_pct:.2f}%)",
            "Avg Loss": f"${self.avg_loss:.2f} ({self.avg_loss_pct:.2f}%)",
            "Largest Win": f"${self.largest_win:.2f}",
            "Largest Loss": f"${self.largest_loss:.2f}",
            "Avg Trade Duration": f"{self.avg_trade_duration:.1f} days",
            "Avg MAE": f"${self.avg_mae:.2f}",
            "Avg MFE": f"${self.avg_mfe:.2f}",
        }


class Backtester:
    """
    Advanced backtesting engine for trading strategies

    Features:
    - Full position management
    - Commission and slippage modeling
    - Advanced risk metrics
    - Portfolio-level backtesting
    - Walk-forward optimization support
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        """Reset backtester state"""
        self.cash = self.config.initial_capital
        self.equity = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.current_date = None

    def calculate_commission(self, price: float, shares: int) -> float:
        """Calculate commission for a trade"""
        commission = price * shares * self.config.commission_pct
        return max(commission, self.config.commission_min)

    def calculate_slippage(self, price: float, shares: int) -> float:
        """Calculate slippage for a trade"""
        return price * shares * self.config.slippage_pct

    def calculate_position_size(self, price: float) -> int:
        """Calculate number of shares based on position sizing"""
        available_capital = self.equity * self.config.position_size / max(1, self.config.max_positions)
        shares = int(available_capital / (price * self.config.margin_requirement))
        return max(0, shares)

    def enter_long(self, symbol: str, date: datetime, price: float, shares: int = None) -> bool:
        """Enter a long position"""
        if symbol in self.positions:
            return False  # Already have position

        if len(self.positions) >= self.config.max_positions:
            return False  # Max positions reached

        # Calculate shares if not provided
        if shares is None:
            shares = self.calculate_position_size(price)

        if shares <= 0:
            return False

        # Calculate costs
        commission = self.calculate_commission(price, shares)
        slippage = self.calculate_slippage(price, shares)
        total_cost = price * shares + commission + slippage

        # Check if we have enough cash
        if total_cost > self.cash:
            # Reduce shares to fit available cash
            shares = int((self.cash - commission - slippage) / price)
            if shares <= 0:
                return False
            total_cost = price * shares + commission + slippage

        # Update cash
        self.cash -= total_cost

        # Create position
        position = Position(
            symbol=symbol,
            side=PositionSide.LONG,
            shares=shares,
            entry_price=price,
            entry_date=date,
            current_price=price
        )
        self.positions[symbol] = position

        # Create trade
        trade = Trade(
            entry_date=date,
            entry_price=price,
            shares=shares,
            side=PositionSide.LONG,
            commission=commission,
            slippage=slippage
        )
        self.trades.append(trade)

        return True

    def enter_short(self, symbol: str, date: datetime, price: float, shares: int = None) -> bool:
        """Enter a short position"""
        if symbol in self.positions:
            return False  # Already have position

        if len(self.positions) >= self.config.max_positions:
            return False  # Max positions reached

        # Calculate shares if not provided
        if shares is None:
            shares = self.calculate_position_size(price)

        if shares <= 0:
            return False

        # Calculate costs
        commission = self.calculate_commission(price, shares)
        slippage = self.calculate_slippage(price, shares)
        margin_required = price * shares * self.config.margin_requirement

        # Check if we have enough cash for margin
        if margin_required + commission + slippage > self.cash:
            # Reduce shares to fit available cash
            shares = int((self.cash - commission - slippage) / (price * self.config.margin_requirement))
            if shares <= 0:
                return False
            margin_required = price * shares * self.config.margin_requirement

        # Update cash (receive proceeds from short sale, minus margin and costs)
        self.cash += (price * shares - margin_required - commission - slippage)

        # Create position
        position = Position(
            symbol=symbol,
            side=PositionSide.SHORT,
            shares=shares,
            entry_price=price,
            entry_date=date,
            current_price=price
        )
        self.positions[symbol] = position

        # Create trade
        trade = Trade(
            entry_date=date,
            entry_price=price,
            shares=shares,
            side=PositionSide.SHORT,
            commission=commission,
            slippage=slippage
        )
        self.trades.append(trade)

        return True

    def exit_position(self, symbol: str, date: datetime, price: float) -> bool:
        """Exit a position"""
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]

        # Calculate costs
        commission = self.calculate_commission(price, position.shares)
        slippage = self.calculate_slippage(price, position.shares)

        # Update cash
        if position.side == PositionSide.LONG:
            proceeds = price * position.shares - commission - slippage
            self.cash += proceeds
        else:  # SHORT
            cost = price * position.shares + commission + slippage
            margin_returned = position.entry_price * position.shares * self.config.margin_requirement
            self.cash += (margin_returned - cost)

        # Close the trade
        trade = self.trades[-1]  # Get most recent trade (should be this position)
        trade.close(date, price, commission, slippage)
        trade.mae = position.mae
        trade.mfe = position.mfe

        # Remove position
        del self.positions[symbol]

        return True

    def update(self, date: datetime, prices: Dict[str, float]):
        """Update backtester with current prices"""
        self.current_date = date

        # Update all positions
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update(prices[symbol])

        # Calculate current equity
        position_value = sum(
            pos.shares * pos.current_price if pos.side == PositionSide.LONG
            else pos.entry_price * pos.shares - (pos.current_price - pos.entry_price) * pos.shares
            for pos in self.positions.values()
        )
        self.equity = self.cash + position_value

        # Record equity curve
        self.equity_curve.append({
            'date': date,
            'equity': self.equity,
            'cash': self.cash,
            'positions': len(self.positions)
        })

    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        benchmark_data: pd.DataFrame = None,
        symbol: str = 'symbol'
    ) -> BacktestResults:
        """
        Run a backtest with the given strategy function

        Args:
            data: DataFrame with OHLCV data
            strategy_func: Function that takes (backtester, date, row) and generates signals
            benchmark_data: Optional benchmark data for comparison
            symbol: Symbol name for position tracking

        Returns:
            BacktestResults object with performance metrics
        """
        self.reset()

        # Run strategy on each bar
        for idx, row in data.iterrows():
            date = idx if isinstance(idx, datetime) else pd.to_datetime(idx)

            # Update positions with current prices (use actual symbol, not hardcoded 'symbol')
            self.update(date, {symbol: row['Close']})

            # Run strategy logic
            strategy_func(self, date, row)

        # Close any remaining positions at final price
        final_date = data.index[-1]
        final_price = data.iloc[-1]['Close']
        for sym in list(self.positions.keys()):
            self.exit_position(sym, final_date, final_price)

        # Update equity curve one final time after closing all positions
        # This ensures the final equity reflects exit commissions/slippage
        if len(self.positions) == 0:  # All positions closed
            self.equity = self.cash  # No position value, just cash
            self.equity_curve.append({
                'date': final_date,
                'equity': self.equity,
                'cash': self.cash,
                'positions': 0
            })

        # Calculate results
        return self.calculate_results(data, benchmark_data)

    def calculate_results(
        self,
        data: pd.DataFrame,
        benchmark_data: pd.DataFrame = None
    ) -> BacktestResults:
        """Calculate backtest results and metrics"""
        results = BacktestResults()

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) == 0:
            return results

        equity_df.set_index('date', inplace=True)
        results.equity_curve = equity_df

        # Calculate returns
        initial_capital = self.config.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        results.total_return = final_equity - initial_capital
        results.total_return_pct = (results.total_return / initial_capital) * 100

        # Calculate annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365.25
        if years > 0:
            results.annualized_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100

        # Calculate daily returns
        equity_df['returns'] = equity_df['equity'].pct_change()

        # Volatility (annualized)
        results.volatility = equity_df['returns'].std() * np.sqrt(252) * 100

        # Sharpe Ratio
        excess_returns = equity_df['returns'] - (self.config.risk_free_rate / 252)
        if excess_returns.std() > 0:
            results.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        # Sortino Ratio (using downside deviation)
        downside_returns = equity_df['returns'][equity_df['returns'] < 0]
        results.downside_deviation = downside_returns.std() * np.sqrt(252) * 100
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            results.sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()

        # Maximum Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        results.max_drawdown = abs(equity_df['drawdown'].min())

        # Maximum Drawdown Duration
        in_drawdown = equity_df['drawdown'] < 0
        drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()
        if in_drawdown.any():
            max_dd_period = drawdown_periods[in_drawdown].value_counts().max()
            results.max_drawdown_duration = max_dd_period

        # Calmar Ratio
        if results.max_drawdown > 0:
            results.calmar_ratio = results.annualized_return / results.max_drawdown

        # Benchmark comparison
        if benchmark_data is not None and len(benchmark_data) > 0:
            benchmark_return = ((benchmark_data.iloc[-1]['Close'] / benchmark_data.iloc[0]['Close']) - 1) * 100
            results.benchmark_return = benchmark_return
            results.alpha = results.total_return_pct - benchmark_return

            # Calculate beta (correlation to benchmark)
            if len(benchmark_data) == len(equity_df):
                benchmark_returns = benchmark_data['Close'].pct_change()
                covariance = np.cov(equity_df['returns'].dropna(), benchmark_returns.dropna())[0][1]
                benchmark_variance = np.var(benchmark_returns.dropna())
                if benchmark_variance > 0:
                    results.beta = covariance / benchmark_variance

        # Trade statistics
        completed_trades = [t for t in self.trades if t.exit_date is not None]
        results.total_trades = len(completed_trades)
        results.trades = completed_trades

        if results.total_trades > 0:
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            losing_trades = [t for t in completed_trades if t.pnl <= 0]

            results.winning_trades = len(winning_trades)
            results.losing_trades = len(losing_trades)
            results.win_rate = (results.winning_trades / results.total_trades) * 100

            if results.winning_trades > 0:
                results.avg_win = np.mean([t.pnl for t in winning_trades])
                results.avg_win_pct = np.mean([t.pnl_pct for t in winning_trades])
                results.largest_win = max([t.pnl for t in winning_trades])

            if results.losing_trades > 0:
                results.avg_loss = np.mean([t.pnl for t in losing_trades])
                results.avg_loss_pct = np.mean([t.pnl_pct for t in losing_trades])
                results.largest_loss = min([t.pnl for t in losing_trades])

            # Profit Factor
            total_wins = sum([t.pnl for t in winning_trades])
            total_losses = abs(sum([t.pnl for t in losing_trades]))
            if total_losses > 0:
                results.profit_factor = total_wins / total_losses

            # Expectancy
            results.expectancy = np.mean([t.pnl for t in completed_trades])

            # Average trade duration
            durations = [(t.exit_date - t.entry_date).days for t in completed_trades]
            results.avg_trade_duration = np.mean(durations)

            # MAE/MFE
            results.avg_mae = np.mean([t.mae for t in completed_trades])
            results.avg_mfe = np.mean([t.mfe for t in completed_trades])

        return results


def calculate_performance_metrics(equity_curve: pd.Series, benchmark: pd.Series = None, risk_free_rate: float = 0.02) -> Dict:
    """
    Standalone function to calculate performance metrics from equity curve

    Args:
        equity_curve: Pandas Series of equity values over time
        benchmark: Optional benchmark equity curve
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of performance metrics
    """
    metrics = {}

    # Calculate returns
    returns = equity_curve.pct_change().dropna()

    # Total return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    metrics['Total Return %'] = total_return

    # Annualized return
    days = len(equity_curve)
    years = days / 252  # Trading days
    if years > 0:
        annualized_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1) * 100
        metrics['Annualized Return %'] = annualized_return

    # Volatility
    volatility = returns.std() * np.sqrt(252) * 100
    metrics['Volatility %'] = volatility

    # Sharpe Ratio
    excess_returns = returns - (risk_free_rate / 252)
    if returns.std() > 0:
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
        metrics['Sharpe Ratio'] = sharpe

    # Maximum Drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax * 100
    max_dd = abs(drawdown.min())
    metrics['Max Drawdown %'] = max_dd

    # Calmar Ratio
    if max_dd > 0 and 'Annualized Return %' in metrics:
        calmar = metrics['Annualized Return %'] / max_dd
        metrics['Calmar Ratio'] = calmar

    # Benchmark comparison
    if benchmark is not None:
        benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
        metrics['Benchmark Return %'] = benchmark_return
        metrics['Alpha %'] = total_return - benchmark_return

    return metrics
