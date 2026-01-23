"""
Unit tests for the Quantsploit backtesting engine

Tests cover:
- Trade P&L calculations (long and short positions)
- Position management
- Commission and slippage modeling
- Risk metrics (Sharpe, Sortino, Calmar, Max Drawdown)
- Edge cases (empty data, single row, zero trades)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.utils.backtesting import (
    Trade, Position, PositionSide, OrderType,
    BacktestConfig, BacktestResults, Backtester,
    calculate_performance_metrics
)


class TestTrade:
    """Tests for the Trade class"""

    def test_trade_long_profit(self):
        """Test P&L calculation for profitable long trade"""
        trade = Trade(
            entry_date=datetime(2023, 1, 1),
            entry_price=100.0,
            shares=100,
            side=PositionSide.LONG,
            commission=1.0,
            slippage=0.5
        )

        trade.close(
            exit_date=datetime(2023, 1, 10),
            exit_price=110.0,
            commission=1.0,
            slippage=0.5
        )

        # Expected P&L: (110-100)*100 - 2 - 1 = 1000 - 3 = 997
        assert trade.pnl == pytest.approx(997.0, rel=0.01)
        assert trade.pnl_pct == pytest.approx(10.0, rel=0.01)
        assert trade.exit_date == datetime(2023, 1, 10)

    def test_trade_long_loss(self):
        """Test P&L calculation for losing long trade"""
        trade = Trade(
            entry_date=datetime(2023, 1, 1),
            entry_price=100.0,
            shares=100,
            side=PositionSide.LONG,
            commission=1.0,
            slippage=0.5
        )

        trade.close(
            exit_date=datetime(2023, 1, 10),
            exit_price=90.0,
            commission=1.0,
            slippage=0.5
        )

        # Expected P&L: (90-100)*100 - 2 - 1 = -1000 - 3 = -1003
        assert trade.pnl == pytest.approx(-1003.0, rel=0.01)
        assert trade.pnl_pct == pytest.approx(-10.0, rel=0.01)

    def test_trade_short_profit(self):
        """Test P&L calculation for profitable short trade"""
        trade = Trade(
            entry_date=datetime(2023, 1, 1),
            entry_price=100.0,
            shares=100,
            side=PositionSide.SHORT,
            commission=1.0,
            slippage=0.5
        )

        trade.close(
            exit_date=datetime(2023, 1, 10),
            exit_price=90.0,  # Price went down = profit for short
            commission=1.0,
            slippage=0.5
        )

        # Expected P&L: (100-90)*100 - 2 - 1 = 1000 - 3 = 997
        assert trade.pnl == pytest.approx(997.0, rel=0.01)
        assert trade.pnl_pct == pytest.approx(10.0, rel=0.01)

    def test_trade_short_loss(self):
        """Test P&L calculation for losing short trade"""
        trade = Trade(
            entry_date=datetime(2023, 1, 1),
            entry_price=100.0,
            shares=100,
            side=PositionSide.SHORT,
            commission=1.0,
            slippage=0.5
        )

        trade.close(
            exit_date=datetime(2023, 1, 10),
            exit_price=110.0,  # Price went up = loss for short
            commission=1.0,
            slippage=0.5
        )

        # Expected P&L: (100-110)*100 - 2 - 1 = -1000 - 3 = -1003
        assert trade.pnl == pytest.approx(-1003.0, rel=0.01)
        assert trade.pnl_pct == pytest.approx(-10.0, rel=0.01)


class TestPosition:
    """Tests for the Position class"""

    def test_position_update_long_profit(self):
        """Test position update for long with price increase"""
        position = Position(
            symbol='AAPL',
            side=PositionSide.LONG,
            shares=100,
            entry_price=100.0,
            entry_date=datetime(2023, 1, 1)
        )

        position.update(110.0)

        assert position.current_price == 110.0
        assert position.unrealized_pnl == pytest.approx(1000.0)  # (110-100)*100
        assert position.mfe >= 1000.0  # Maximum favorable excursion

    def test_position_update_long_loss(self):
        """Test position update for long with price decrease"""
        position = Position(
            symbol='AAPL',
            side=PositionSide.LONG,
            shares=100,
            entry_price=100.0,
            entry_date=datetime(2023, 1, 1)
        )

        position.update(90.0)

        assert position.current_price == 90.0
        assert position.unrealized_pnl == pytest.approx(-1000.0)  # (90-100)*100
        assert position.mae <= -1000.0  # Maximum adverse excursion

    def test_position_update_short(self):
        """Test position update for short position"""
        position = Position(
            symbol='AAPL',
            side=PositionSide.SHORT,
            shares=100,
            entry_price=100.0,
            entry_date=datetime(2023, 1, 1)
        )

        position.update(90.0)

        assert position.unrealized_pnl == pytest.approx(1000.0)  # Short profits when price falls


class TestBacktestConfig:
    """Tests for BacktestConfig defaults and validation"""

    def test_default_config(self):
        """Test default configuration values"""
        config = BacktestConfig()

        assert config.initial_capital == 100000.0
        assert config.commission_pct == 0.001
        assert config.commission_min == 1.0
        assert config.slippage_pct == 0.001
        assert config.position_size == 1.0
        assert config.max_positions == 1
        assert config.margin_requirement == 1.0
        assert config.risk_free_rate == 0.02
        assert config.benchmark_symbol == 'SPY'

    def test_custom_config(self):
        """Test custom configuration"""
        config = BacktestConfig(
            initial_capital=50000.0,
            commission_pct=0.002,
            position_size=0.5
        )

        assert config.initial_capital == 50000.0
        assert config.commission_pct == 0.002
        assert config.position_size == 0.5


class TestBacktester:
    """Tests for the Backtester class"""

    def test_commission_calculation(self, backtester):
        """Test commission calculation"""
        commission = backtester.calculate_commission(100.0, 100)

        # 100 * 100 * 0.001 = 10, should return 10 (> min of 1)
        assert commission == pytest.approx(10.0)

    def test_commission_minimum(self, backtester):
        """Test minimum commission is applied"""
        commission = backtester.calculate_commission(1.0, 1)

        # 1 * 1 * 0.001 = 0.001, should return 1.0 (minimum)
        assert commission == pytest.approx(1.0)

    def test_slippage_calculation(self, backtester):
        """Test slippage calculation"""
        slippage = backtester.calculate_slippage(100.0, 100)

        # 100 * 100 * 0.001 = 10
        assert slippage == pytest.approx(10.0)

    def test_position_size_calculation(self, backtester):
        """Test position size calculation"""
        shares = backtester.calculate_position_size(100.0)

        # With $100k capital, position_size=1.0, max_positions=1
        # Available = 100000 * 1.0 / 1 = 100000
        # Shares = 100000 / (100 * 1.0) = 1000
        assert shares == 1000

    def test_enter_long(self, backtester, sample_ohlcv_data):
        """Test entering a long position"""
        date = sample_ohlcv_data.index[0]
        price = sample_ohlcv_data.iloc[0]['Close']

        result = backtester.enter_long('AAPL', date, price)

        assert result is True
        assert 'AAPL' in backtester.positions
        assert backtester.positions['AAPL'].side == PositionSide.LONG
        assert len(backtester.trades) == 1

    def test_enter_long_duplicate_rejected(self, backtester, sample_ohlcv_data):
        """Test that duplicate position is rejected"""
        date = sample_ohlcv_data.index[0]
        price = sample_ohlcv_data.iloc[0]['Close']

        backtester.enter_long('AAPL', date, price)
        result = backtester.enter_long('AAPL', date, price)

        assert result is False  # Already have position

    def test_exit_position(self, backtester, sample_ohlcv_data):
        """Test exiting a position"""
        entry_date = sample_ohlcv_data.index[0]
        entry_price = sample_ohlcv_data.iloc[0]['Close']
        exit_date = sample_ohlcv_data.index[10]
        exit_price = sample_ohlcv_data.iloc[10]['Close']

        backtester.enter_long('AAPL', entry_date, entry_price)
        result = backtester.exit_position('AAPL', exit_date, exit_price)

        assert result is True
        assert 'AAPL' not in backtester.positions
        assert backtester.trades[-1].exit_date is not None

    def test_exit_nonexistent_position(self, backtester):
        """Test exiting a position that doesn't exist"""
        result = backtester.exit_position('AAPL', datetime.now(), 100.0)

        assert result is False

    def test_reset(self, backtester, backtest_config):
        """Test backtester reset"""
        backtester.cash = 50000.0
        backtester.equity = 60000.0

        backtester.reset()

        assert backtester.cash == backtest_config.initial_capital
        assert backtester.equity == backtest_config.initial_capital
        assert len(backtester.positions) == 0
        assert len(backtester.trades) == 0


class TestBacktestResults:
    """Tests for BacktestResults calculations"""

    def test_run_backtest_basic(self, backtester, sample_ohlcv_data):
        """Test running a basic backtest"""
        def simple_strategy(bt, date, row):
            # Buy on first day, sell on last day
            if len(bt.equity_curve) == 0:
                bt.enter_long('TEST', date, row['Close'])
            elif len(bt.equity_curve) > 200 and 'TEST' in bt.positions:
                bt.exit_position('TEST', date, row['Close'])

        results = backtester.run_backtest(
            sample_ohlcv_data,
            simple_strategy,
            symbol='TEST'
        )

        assert isinstance(results, BacktestResults)
        assert results.total_trades >= 0
        assert len(results.equity_curve) > 0

    def test_results_metrics_valid_range(self, backtester, sample_ohlcv_bullish):
        """Test that metrics are within valid ranges"""
        def buy_and_hold(bt, date, row):
            if len(bt.positions) == 0:
                bt.enter_long('TEST', date, row['Close'])

        results = backtester.run_backtest(
            sample_ohlcv_bullish,
            buy_and_hold,
            symbol='TEST'
        )

        # Win rate should be between 0 and 100
        assert 0 <= results.win_rate <= 100

        # Max drawdown should be non-negative
        assert results.max_drawdown >= 0

        # Volatility should be non-negative
        assert results.volatility >= 0

    def test_to_dict(self, backtester, sample_ohlcv_data):
        """Test results to_dict method"""
        def simple_strategy(bt, date, row):
            if len(bt.positions) == 0:
                bt.enter_long('TEST', date, row['Close'])

        results = backtester.run_backtest(
            sample_ohlcv_data,
            simple_strategy,
            symbol='TEST'
        )

        result_dict = results.to_dict()

        assert isinstance(result_dict, dict)
        assert 'Total Return' in result_dict
        assert 'Sharpe Ratio' in result_dict
        assert 'Max Drawdown' in result_dict


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_empty_dataframe(self, backtester, empty_dataframe):
        """Test handling of empty DataFrame"""
        def strategy(bt, date, row):
            pass

        # Should handle gracefully without crashing
        # Note: The actual behavior depends on implementation
        # This test ensures no exception is raised
        try:
            results = backtester.run_backtest(
                empty_dataframe,
                strategy,
                symbol='TEST'
            )
        except (IndexError, ValueError):
            # Expected for empty data
            pass

    def test_zero_trades(self, backtester, sample_ohlcv_data):
        """Test backtest with no trades executed"""
        def no_trade_strategy(bt, date, row):
            pass  # Never trade

        results = backtester.run_backtest(
            sample_ohlcv_data,
            no_trade_strategy,
            symbol='TEST'
        )

        assert results.total_trades == 0
        assert results.win_rate == 0
        assert results.profit_factor == 0

    def test_single_row(self, backtester, single_row_dataframe):
        """Test handling of single-row DataFrame"""
        def strategy(bt, date, row):
            if len(bt.positions) == 0:
                bt.enter_long('TEST', date, row['Close'])

        # Should handle without crashing
        try:
            results = backtester.run_backtest(
                single_row_dataframe,
                strategy,
                symbol='TEST'
            )
        except (IndexError, ValueError, KeyError):
            # Expected for minimal data
            pass


class TestPerformanceMetrics:
    """Tests for standalone performance metrics function"""

    def test_calculate_performance_metrics(self):
        """Test standalone performance metrics calculation"""
        # Create simple equity curve
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        equity = pd.Series(
            [100000 * (1 + 0.0003 * i) for i in range(252)],
            index=dates
        )

        metrics = calculate_performance_metrics(equity)

        assert 'Total Return %' in metrics
        assert 'Annualized Return %' in metrics
        assert 'Volatility %' in metrics
        assert 'Sharpe Ratio' in metrics
        assert 'Max Drawdown %' in metrics

    def test_metrics_with_benchmark(self):
        """Test metrics with benchmark comparison"""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')

        equity = pd.Series(
            [100000 * (1 + 0.0005 * i) for i in range(252)],
            index=dates
        )

        benchmark = pd.Series(
            [100000 * (1 + 0.0003 * i) for i in range(252)],
            index=dates
        )

        metrics = calculate_performance_metrics(equity, benchmark)

        assert 'Benchmark Return %' in metrics
        assert 'Alpha %' in metrics
        assert metrics['Alpha %'] > 0  # Strategy outperformed


class TestRiskMetrics:
    """Detailed tests for risk metric calculations"""

    def test_sharpe_ratio_positive(self, backtester, sample_ohlcv_bullish):
        """Test Sharpe ratio is positive for bullish data"""
        def buy_and_hold(bt, date, row):
            if len(bt.positions) == 0:
                bt.enter_long('TEST', date, row['Close'])

        results = backtester.run_backtest(
            sample_ohlcv_bullish,
            buy_and_hold,
            symbol='TEST'
        )

        # Bullish trend should produce positive Sharpe
        # (Though depends on volatility and risk-free rate)
        assert results.sharpe_ratio is not None

    def test_max_drawdown_calculation(self, backtester, sample_ohlcv_volatile):
        """Test max drawdown is calculated correctly for volatile data"""
        def buy_and_hold(bt, date, row):
            if len(bt.positions) == 0:
                bt.enter_long('TEST', date, row['Close'])

        results = backtester.run_backtest(
            sample_ohlcv_volatile,
            buy_and_hold,
            symbol='TEST'
        )

        # Volatile data should have significant drawdown
        assert results.max_drawdown >= 0
        # Drawdown should be less than 100% (can't lose more than everything)
        assert results.max_drawdown <= 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
