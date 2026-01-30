"""
Integration tests for key user workflows

Tests cover:
- End-to-end module execution workflow
- Module loading and configuration
- Backtest execution pipeline
- Data fetching and analysis pipeline
- UI command workflows
- Database operations

★ Insight ─────────────────────────────────────
Integration tests verify that modules work together correctly.
Unlike unit tests that isolate components, these tests exercise
the real interaction patterns that users experience in the app.
─────────────────────────────────────────────────
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Fixtures for generating realistic test data
@pytest.fixture
def sample_price_data():
    """Generate realistic price data for testing"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')

    # Generate random walk prices
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'High': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'Low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    # Ensure High is highest, Low is lowest
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

    return df


@pytest.fixture
def mock_data_fetcher(sample_price_data):
    """Create a mock data fetcher that returns sample data"""
    with patch('quantsploit.utils.data_fetcher.DataFetcher') as MockFetcher:
        mock_fetcher = Mock()
        mock_fetcher.get_stock_data.return_value = sample_price_data
        mock_fetcher.get_realtime_quote.return_value = {
            'symbol': 'AAPL',
            'price': sample_price_data['Close'].iloc[-1],
            'change': 1.5,
            'change_percent': 0.75
        }
        MockFetcher.return_value = mock_fetcher
        yield mock_fetcher


class TestBacktestingWorkflow:
    """Integration tests for backtesting workflow"""

    def test_backtest_config_creation(self, sample_price_data):
        """Test BacktestConfig creation with correct parameters"""
        from quantsploit.utils.backtesting import BacktestConfig

        # Configure backtest with correct parameter names
        config = BacktestConfig(
            initial_capital=10000,
            commission_pct=0.001,
            slippage_pct=0.0005
        )

        assert config.initial_capital == 10000
        assert config.commission_pct == 0.001
        assert config.slippage_pct == 0.0005

    def test_backtester_initialization(self, sample_price_data):
        """Test Backtester initialization"""
        from quantsploit.utils.backtesting import BacktestConfig, Backtester

        config = BacktestConfig(initial_capital=10000)
        backtester = Backtester(config)

        assert backtester.config.initial_capital == 10000

    def test_position_creation(self, sample_price_data):
        """Test Position creation with correct parameters"""
        from quantsploit.utils.backtesting import Position, PositionSide

        # Create a position using correct field names
        entry_price = sample_price_data['Close'].iloc[0]
        position = Position(
            symbol='AAPL',
            shares=100,
            entry_price=entry_price,
            entry_date=sample_price_data.index[0],
            side=PositionSide.LONG
        )

        assert position.symbol == 'AAPL'
        assert position.shares == 100
        assert position.side == PositionSide.LONG

    def test_position_update(self, sample_price_data):
        """Test Position update tracking"""
        from quantsploit.utils.backtesting import Position, PositionSide

        entry_price = sample_price_data['Close'].iloc[0]
        position = Position(
            symbol='AAPL',
            shares=100,
            entry_price=entry_price,
            entry_date=sample_price_data.index[0],
            side=PositionSide.LONG
        )

        # Update position
        new_price = entry_price * 1.05  # 5% gain
        position.update(new_price)

        assert position.current_price == new_price
        assert position.unrealized_pnl > 0


class TestDataPipelineWorkflow:
    """Integration tests for data pipeline"""

    def test_data_structure_validation(self, sample_price_data):
        """Test data flows through validation correctly"""
        # Verify data structure is valid for analysis
        assert 'Open' in sample_price_data.columns
        assert 'High' in sample_price_data.columns
        assert 'Low' in sample_price_data.columns
        assert 'Close' in sample_price_data.columns
        assert 'Volume' in sample_price_data.columns

    def test_data_integrity(self, sample_price_data):
        """Test data integrity constraints"""
        # Verify data integrity
        assert (sample_price_data['High'] >= sample_price_data['Low']).all()
        assert (sample_price_data['High'] >= sample_price_data['Close']).all()
        assert (sample_price_data['Close'] >= sample_price_data['Low']).all()

    def test_technical_indicator_calculation(self, sample_price_data):
        """Test technical indicators calculate correctly"""
        from quantsploit.utils.ta_compat import sma, ema, rsi, macd

        # Calculate indicators using lowercase function names
        sma_10 = sma(sample_price_data['Close'], 10)
        ema_10 = ema(sample_price_data['Close'], 10)
        rsi_14 = rsi(sample_price_data['Close'], 14)
        macd_result = macd(sample_price_data['Close'])

        # Verify output shapes
        assert len(sma_10) == len(sample_price_data)
        assert len(ema_10) == len(sample_price_data)
        assert len(rsi_14) == len(sample_price_data)

        # Verify RSI bounds
        valid_rsi = rsi_14[~np.isnan(rsi_14)]
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()


class TestUIIntegrationWorkflow:
    """Integration tests for UI components working together"""

    def test_command_handler_to_display_workflow(self):
        """Test command handler output goes to display"""
        from quantsploit.ui.commands import CommandHandler

        # Mock framework
        mock_framework = Mock()
        mock_framework.session = Mock()
        mock_framework.session.current_module = None
        mock_framework.session.add_command = Mock()
        mock_framework.modules = {}
        mock_framework.list_modules = Mock(return_value=[])
        mock_framework.database = Mock()
        mock_framework.database.get_watchlist = Mock(return_value=[])

        # Create handler (which creates its own display)
        with patch('quantsploit.ui.commands.Display') as MockDisplay:
            mock_display = Mock()
            MockDisplay.return_value = mock_display

            handler = CommandHandler(mock_framework)

            # Execute a command
            handler.execute('help')

            # Verify display was called
            mock_display.print_help.assert_called()

    def test_show_modules_workflow(self):
        """Test showing modules workflow"""
        from quantsploit.ui.commands import CommandHandler

        # Mock framework with modules
        mock_framework = Mock()
        mock_framework.session = Mock()
        mock_framework.session.current_module = None
        mock_framework.session.add_command = Mock()
        mock_framework.list_modules = Mock(return_value=[
            Mock(path='strategies/sma', category='strategies', description='SMA strategy'),
            Mock(path='analysis/stock', category='analysis', description='Stock analysis')
        ])

        with patch('quantsploit.ui.commands.Display') as MockDisplay:
            mock_display = Mock()
            MockDisplay.return_value = mock_display

            handler = CommandHandler(mock_framework)
            handler.execute('show modules')

            # Verify modules were displayed
            mock_display.print_modules.assert_called()


class TestModuleLifecycleWorkflow:
    """Integration tests for module lifecycle"""

    def test_module_use_set_run_back_workflow(self):
        """Test complete module lifecycle: use -> set -> run -> back"""
        from quantsploit.ui.commands import CommandHandler

        # Create mock module
        mock_module = Mock()
        mock_module.name = 'Test Strategy'
        mock_module.description = 'Test description'
        mock_module.show_options = Mock(return_value={
            'SYMBOL': {'required': True, 'value': '', 'description': 'Symbol'}
        })
        mock_module.set_option = Mock(return_value=True)
        mock_module.show_info = Mock(return_value={
            'name': 'Test',
            'category': 'test',
            'description': 'Test',
            'author': 'Tester',
            'options': {}
        })

        # Mock framework
        mock_framework = Mock()
        mock_framework.session = Mock()
        mock_framework.session.current_module = None
        mock_framework.session.add_command = Mock()
        mock_framework.session.unload_module = Mock()
        mock_framework.use_module = Mock(return_value=mock_module)
        mock_framework.run_module = Mock(return_value={'success': True, 'return': 10.5})
        mock_framework.search_modules = Mock(return_value=[])

        with patch('quantsploit.ui.commands.Display') as MockDisplay:
            mock_display = Mock()
            MockDisplay.return_value = mock_display

            handler = CommandHandler(mock_framework)

            # Step 1: Use module
            handler.execute('use strategies/test')
            mock_framework.use_module.assert_called()

            # Update session state for subsequent commands
            mock_framework.session.current_module = mock_module

            # Step 2: Set option
            handler.execute('set SYMBOL AAPL')
            mock_module.set_option.assert_called_with('SYMBOL', 'AAPL')

            # Step 3: Run
            handler.execute('run')
            mock_framework.run_module.assert_called_with(mock_module)

            # Step 4: Back
            handler.execute('back')
            mock_framework.session.unload_module.assert_called()


class TestStatisticalAnalysisWorkflow:
    """Integration tests for statistical analysis"""

    def test_statistical_analyzer_robust_stats(self, sample_price_data):
        """Test statistical analyzer robust stats calculation"""
        from quantsploit.utils.statistical_analyzer import StatisticalAnalyzer

        # Create returns series
        returns = sample_price_data['Close'].pct_change().dropna()

        # Analyze with correct method name
        analyzer = StatisticalAnalyzer()
        stats = analyzer.calculate_robust_stats(returns, 'returns')

        # Verify statistical outputs
        assert stats is not None
        assert hasattr(stats, 'mean') or hasattr(stats, 'median')


class TestDatabaseWorkflow:
    """Integration tests for database operations"""

    def test_watchlist_crud_operations(self, tmp_path):
        """Test complete watchlist CRUD workflow"""
        from quantsploit.core.database import Database

        # Create database in temp directory
        db = Database(str(tmp_path / 'test.db'))

        # Create
        result = db.add_to_watchlist('AAPL', 'Tech leader')
        assert result is True

        # Read
        watchlist = db.get_watchlist()
        assert len(watchlist) >= 1
        assert any(w['symbol'] == 'AAPL' for w in watchlist)

        # Delete
        db.remove_from_watchlist('AAPL')
        watchlist = db.get_watchlist()
        assert not any(w['symbol'] == 'AAPL' for w in watchlist)

    def test_multiple_watchlist_entries(self, tmp_path):
        """Test adding multiple items to watchlist"""
        from quantsploit.core.database import Database

        db = Database(str(tmp_path / 'test.db'))

        # Add multiple symbols
        db.add_to_watchlist('AAPL', 'Apple Inc')
        db.add_to_watchlist('MSFT', 'Microsoft')
        db.add_to_watchlist('GOOGL', 'Google')

        watchlist = db.get_watchlist()
        symbols = [w['symbol'] for w in watchlist]

        assert 'AAPL' in symbols
        assert 'MSFT' in symbols
        assert 'GOOGL' in symbols


class TestCoreFrameworkWorkflow:
    """Integration tests for core framework"""

    def test_session_management(self):
        """Test session management"""
        from quantsploit.core.session import Session

        session = Session()

        # Add commands to history
        session.add_command('help')
        session.add_command('show modules')
        session.add_command('use strategies/sma')

        # Verify history
        assert len(session.command_history) == 3

    def test_session_export(self):
        """Test session export"""
        from quantsploit.core.session import Session

        session = Session()
        session.add_command('help')

        exported = session.export_session()
        assert exported is not None
        assert 'start_time' in exported or 'command_count' in exported or len(exported) > 0


class TestEndToEndWorkflow:
    """Complete end-to-end workflow tests"""

    def test_technical_analysis_to_display(self, sample_price_data):
        """Test technical analysis to display workflow"""
        from quantsploit.utils.ta_compat import sma, rsi
        from quantsploit.ui.display import Display

        # Calculate technical indicators
        sma_10 = sma(sample_price_data['Close'], 10)
        sma_30 = sma(sample_price_data['Close'], 30)
        rsi_14 = rsi(sample_price_data['Close'], 14)

        # Generate signals
        signals = np.where(sma_10 > sma_30, 1, -1)

        # Calculate returns based on signals
        returns = sample_price_data['Close'].pct_change() * pd.Series(signals).shift(1)
        returns = returns.dropna()

        # Create summary stats
        stats = {
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'total_signals': len(signals)
        }

        # Create display mock and format output
        with patch('quantsploit.ui.display.RichConsole') as MockConsole:
            mock_console = Mock()
            MockConsole.return_value = mock_console

            display = Display()
            display.console = mock_console

            # Display results
            display._print_dict(stats, "Strategy Statistics")
            assert mock_console.print.called

    def test_full_command_session(self):
        """Test a full command session workflow"""
        from quantsploit.ui.commands import CommandHandler

        # Setup mock framework
        mock_framework = Mock()
        mock_framework.session = Mock()
        mock_framework.session.current_module = None
        mock_framework.session.add_command = Mock()
        mock_framework.session.unload_module = Mock()
        mock_framework.session.command_history = []
        mock_framework.modules = {}
        mock_framework.list_modules = Mock(return_value=[])
        mock_framework.database = Mock()
        mock_framework.database.get_watchlist = Mock(return_value=[])
        mock_framework.search_modules = Mock(return_value=[])

        with patch('quantsploit.ui.commands.Display') as MockDisplay:
            mock_display = Mock()
            MockDisplay.return_value = mock_display

            handler = CommandHandler(mock_framework)

            # Simulate a user session
            commands = [
                'help',
                'show modules',
                'search sma',
            ]

            for cmd in commands:
                result = handler.execute(cmd)
                assert result is True  # Should not exit

            # Exit command should return False
            result = handler.execute('exit')
            assert result is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
