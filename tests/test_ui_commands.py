"""
Unit tests for UI Commands Module

Tests cover:
- Command registration and routing
- Module loading workflow (use, back, info)
- Option management (set, unset, options)
- Module execution (run)
- Search functionality
- Watchlist management
- Quote fetching
- Session management
- Input validation and error handling

★ Insight ─────────────────────────────────────
This test file validates the Metasploit-inspired command architecture.
The workflow pattern use → set → run is core to the application's UX.
Testing command routing ensures users get consistent feedback.
─────────────────────────────────────────────────
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.ui.commands import CommandHandler


@pytest.fixture
def mock_module():
    """Create a mock module with standard options"""
    module = Mock()
    module.name = 'SMA Crossover'
    module.description = 'Simple moving average crossover strategy'
    module.show_options = Mock(return_value={
        'SYMBOL': {'required': True, 'default': '', 'value': '', 'description': 'Stock ticker'},
        'FAST_PERIOD': {'required': False, 'default': 10, 'value': 10, 'description': 'Fast SMA period'},
        'SLOW_PERIOD': {'required': False, 'default': 50, 'value': 50, 'description': 'Slow SMA period'}
    })
    module.show_info = Mock(return_value={
        'name': 'SMA Crossover',
        'description': 'Simple moving average crossover strategy',
        'author': 'Quantsploit',
        'options': module.show_options()
    })
    module.set_option = Mock(return_value=True)
    return module


@pytest.fixture
def mock_session():
    """Create a mock session object"""
    session = Mock()
    session.current_module = None
    session.command_history = []
    session.add_command = Mock()
    session.unload_module = Mock()
    session.export_session = Mock(return_value={
        'start_time': datetime.now().isoformat(),
        'commands_run': 5,
        'current_module': None
    })
    return session


@pytest.fixture
def mock_database():
    """Create a mock database object"""
    db = Mock()
    db.get_watchlist = Mock(return_value=[
        {'symbol': 'AAPL', 'notes': 'Tech giant'},
        {'symbol': 'MSFT', 'notes': 'Cloud leader'}
    ])
    db.add_to_watchlist = Mock(return_value=True)
    db.remove_from_watchlist = Mock(return_value=True)
    return db


@pytest.fixture
def mock_framework(mock_session, mock_database):
    """Create a mock framework object"""
    framework = Mock()
    framework.session = mock_session
    framework.database = mock_database
    framework.list_modules = Mock(return_value=[
        {'path': 'strategies/sma_crossover', 'name': 'SMA Crossover', 'description': 'SMA strategy'},
        {'path': 'analysis/stock_analyzer', 'name': 'Stock Analyzer', 'description': 'Analyze stocks'}
    ])
    framework.use_module = Mock(return_value=None)
    framework.search_modules = Mock(return_value=[])
    framework.run_module = Mock(return_value={'status': 'success', 'total_return': 15.5})
    return framework


@pytest.fixture
def command_handler(mock_framework):
    """Create a CommandHandler instance with mock framework"""
    with patch('quantsploit.ui.commands.Display') as MockDisplay:
        # Create mock display instance
        mock_display = Mock()
        MockDisplay.return_value = mock_display

        handler = CommandHandler(mock_framework)
        handler.display = mock_display  # Ensure we can access it for assertions
        return handler


class TestCommandRegistration:
    """Tests for command registration"""

    def test_commands_registered(self, command_handler):
        """Test that commands are properly registered"""
        assert command_handler.commands is not None
        assert len(command_handler.commands) > 0

    def test_help_command_registered(self, command_handler):
        """Test help command is registered"""
        assert 'help' in command_handler.commands
        assert '?' in command_handler.commands

    def test_core_commands_registered(self, command_handler):
        """Test core commands are registered"""
        core_commands = ['show', 'use', 'back', 'info', 'options',
                        'set', 'unset', 'run', 'search', 'exit']
        for cmd in core_commands:
            assert cmd in command_handler.commands, f"Command '{cmd}' not registered"

    def test_exploit_alias_registered(self, command_handler):
        """Test 'exploit' is an alias for 'run'"""
        assert 'exploit' in command_handler.commands
        assert command_handler.commands['exploit'] == command_handler.commands['run']

    def test_quit_alias_registered(self, command_handler):
        """Test 'quit' is an alias for 'exit'"""
        assert 'quit' in command_handler.commands
        assert command_handler.commands['quit'] == command_handler.commands['exit']

    def test_question_mark_alias(self, command_handler):
        """Test '?' is alias for help"""
        assert command_handler.commands['?'] == command_handler.commands['help']


class TestCommandDescriptions:
    """Tests for command descriptions"""

    def test_get_command_descriptions(self, command_handler):
        """Test getting command descriptions"""
        descriptions = command_handler.get_command_descriptions()
        assert isinstance(descriptions, dict)
        assert len(descriptions) > 0

    def test_descriptions_have_content(self, command_handler):
        """Test descriptions are not empty"""
        descriptions = command_handler.get_command_descriptions()
        for cmd, desc in descriptions.items():
            assert len(desc) > 0, f"Command '{cmd}' has empty description"


class TestExecuteCommand:
    """Tests for execute method"""

    def test_execute_empty_command(self, command_handler, mock_framework):
        """Test executing an empty command returns True"""
        result = command_handler.execute('')
        assert result is True

    def test_execute_whitespace_command(self, command_handler, mock_framework):
        """Test executing whitespace-only command returns True"""
        result = command_handler.execute('   ')
        assert result is True

    def test_execute_unknown_command(self, command_handler, mock_framework):
        """Test executing an unknown command"""
        result = command_handler.execute('nonexistent_command')
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_execute_adds_to_history(self, command_handler, mock_framework):
        """Test that commands are added to session history"""
        command_handler.execute('help')
        mock_framework.session.add_command.assert_called_with('help')

    def test_execute_handles_shlex_error(self, command_handler, mock_framework):
        """Test handling of invalid command syntax"""
        result = command_handler.execute('set value "unclosed quote')
        assert result is True
        command_handler.display.print_error.assert_called()


class TestHelpCommand:
    """Tests for help command"""

    def test_help_returns_true(self, command_handler):
        """Test help command returns True"""
        result = command_handler.cmd_help([])
        assert result is True

    def test_help_calls_display(self, command_handler):
        """Test help command calls display.print_help"""
        command_handler.cmd_help([])
        command_handler.display.print_help.assert_called_once()


class TestShowCommand:
    """Tests for show command"""

    def test_show_no_args_shows_error(self, command_handler):
        """Test show command with no arguments shows error"""
        result = command_handler.cmd_show([])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_show_modules(self, command_handler, mock_framework):
        """Test show modules command"""
        result = command_handler.cmd_show(['modules'])
        assert result is True
        mock_framework.list_modules.assert_called()
        command_handler.display.print_modules.assert_called()

    def test_show_modules_with_category(self, command_handler, mock_framework):
        """Test show modules with category filter"""
        result = command_handler.cmd_show(['modules', 'strategies'])
        assert result is True
        mock_framework.list_modules.assert_called_with('strategies')

    def test_show_options_no_module(self, command_handler, mock_framework):
        """Test show options when no module loaded"""
        mock_framework.session.current_module = None
        result = command_handler.cmd_show(['options'])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_show_options_with_module(self, command_handler, mock_framework, mock_module):
        """Test show options when module is loaded"""
        mock_framework.session.current_module = mock_module
        result = command_handler.cmd_show(['options'])
        assert result is True
        mock_module.show_options.assert_called()
        command_handler.display.print_options.assert_called()

    def test_show_watchlist_with_items(self, command_handler, mock_framework):
        """Test show watchlist with items"""
        result = command_handler.cmd_show(['watchlist'])
        assert result is True
        mock_framework.database.get_watchlist.assert_called()

    def test_show_watchlist_empty(self, command_handler, mock_framework):
        """Test show watchlist when empty"""
        mock_framework.database.get_watchlist.return_value = []
        result = command_handler.cmd_show(['watchlist'])
        assert result is True
        command_handler.display.print_info.assert_called()

    def test_show_unknown_option(self, command_handler):
        """Test show with unknown option"""
        result = command_handler.cmd_show(['unknown_option'])
        assert result is True
        command_handler.display.print_error.assert_called()


class TestUseCommand:
    """Tests for use command"""

    def test_use_no_args(self, command_handler):
        """Test use command with no arguments"""
        result = command_handler.cmd_use([])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_use_exact_path_found(self, command_handler, mock_framework, mock_module):
        """Test use command with exact module path"""
        mock_framework.use_module.return_value = mock_module

        result = command_handler.cmd_use(['strategies/sma_crossover'])
        assert result is True
        mock_framework.use_module.assert_called_with('strategies/sma_crossover')
        command_handler.display.print_success.assert_called()

    def test_use_module_not_found_search_empty(self, command_handler, mock_framework):
        """Test use command when module not found and no search results"""
        mock_framework.use_module.return_value = None
        mock_framework.search_modules.return_value = []

        result = command_handler.cmd_use(['nonexistent'])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_use_module_not_found_single_match(self, command_handler, mock_framework, mock_module):
        """Test use command auto-loads when single search match"""
        # First call returns None (exact match fails)
        # Second call returns module (after search finds single match)
        mock_framework.use_module.side_effect = [None, mock_module]
        mock_framework.search_modules.return_value = [{'path': 'strategies/sma_crossover'}]

        result = command_handler.cmd_use(['sma'])
        assert result is True

    def test_use_module_not_found_multiple_matches(self, command_handler, mock_framework):
        """Test use command shows options when multiple matches"""
        mock_framework.use_module.return_value = None
        mock_framework.search_modules.return_value = [
            {'path': 'strategies/sma_crossover'},
            {'path': 'strategies/sma_dual'}
        ]

        result = command_handler.cmd_use(['sma'])
        assert result is True
        command_handler.display.print_modules.assert_called()


class TestBackCommand:
    """Tests for back command"""

    def test_back_with_module_loaded(self, command_handler, mock_framework, mock_module):
        """Test back command when module is loaded"""
        mock_framework.session.current_module = mock_module

        result = command_handler.cmd_back([])
        assert result is True
        mock_framework.session.unload_module.assert_called_once()
        command_handler.display.print_success.assert_called()

    def test_back_no_module_loaded(self, command_handler, mock_framework):
        """Test back command when no module loaded"""
        mock_framework.session.current_module = None

        result = command_handler.cmd_back([])
        assert result is True
        command_handler.display.print_warning.assert_called()


class TestInfoCommand:
    """Tests for info command"""

    def test_info_with_module_loaded(self, command_handler, mock_framework, mock_module):
        """Test info command when module is loaded"""
        mock_framework.session.current_module = mock_module

        result = command_handler.cmd_info([])
        assert result is True
        mock_module.show_info.assert_called_once()
        command_handler.display.print_module_info.assert_called()

    def test_info_no_module_loaded(self, command_handler, mock_framework):
        """Test info command when no module loaded"""
        mock_framework.session.current_module = None

        result = command_handler.cmd_info([])
        assert result is True
        command_handler.display.print_error.assert_called()


class TestOptionsCommand:
    """Tests for options command"""

    def test_options_delegates_to_show(self, command_handler, mock_framework):
        """Test options command delegates to show options"""
        with patch.object(command_handler, 'cmd_show', return_value=True) as mock_show:
            result = command_handler.cmd_options([])
            assert result is True
            mock_show.assert_called_with(['options'])


class TestSetCommand:
    """Tests for set command"""

    def test_set_no_module_loaded(self, command_handler, mock_framework):
        """Test set command when no module loaded"""
        mock_framework.session.current_module = None

        result = command_handler.cmd_set(['SYMBOL', 'AAPL'])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_set_missing_args(self, command_handler, mock_framework, mock_module):
        """Test set command with missing arguments"""
        mock_framework.session.current_module = mock_module

        result = command_handler.cmd_set(['SYMBOL'])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_set_valid_option(self, command_handler, mock_framework, mock_module):
        """Test setting a valid option"""
        mock_framework.session.current_module = mock_module
        mock_module.set_option.return_value = True

        result = command_handler.cmd_set(['SYMBOL', 'AAPL'])
        assert result is True
        mock_module.set_option.assert_called_with('SYMBOL', 'AAPL')
        command_handler.display.print_success.assert_called()

    def test_set_invalid_option(self, command_handler, mock_framework, mock_module):
        """Test setting an invalid option"""
        mock_framework.session.current_module = mock_module
        mock_module.set_option.return_value = False

        result = command_handler.cmd_set(['INVALID', 'value'])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_set_value_with_spaces(self, command_handler, mock_framework, mock_module):
        """Test setting a value with spaces"""
        mock_framework.session.current_module = mock_module
        mock_module.set_option.return_value = True

        result = command_handler.cmd_set(['DESCRIPTION', 'my', 'multi', 'word', 'value'])
        assert result is True
        mock_module.set_option.assert_called_with('DESCRIPTION', 'my multi word value')

    def test_set_converts_to_uppercase(self, command_handler, mock_framework, mock_module):
        """Test that option names are converted to uppercase"""
        mock_framework.session.current_module = mock_module
        mock_module.set_option.return_value = True

        result = command_handler.cmd_set(['symbol', 'AAPL'])
        mock_module.set_option.assert_called_with('SYMBOL', 'AAPL')


class TestUnsetCommand:
    """Tests for unset command"""

    def test_unset_no_module_loaded(self, command_handler, mock_framework):
        """Test unset command when no module loaded"""
        mock_framework.session.current_module = None

        result = command_handler.cmd_unset(['SYMBOL'])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_unset_missing_args(self, command_handler, mock_framework, mock_module):
        """Test unset command with missing arguments"""
        mock_framework.session.current_module = mock_module

        result = command_handler.cmd_unset([])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_unset_valid_option(self, command_handler, mock_framework, mock_module):
        """Test unsetting a valid option"""
        mock_framework.session.current_module = mock_module
        mock_module.set_option.return_value = True

        result = command_handler.cmd_unset(['SYMBOL'])
        assert result is True
        mock_module.set_option.assert_called_with('SYMBOL', None)
        command_handler.display.print_success.assert_called()

    def test_unset_invalid_option(self, command_handler, mock_framework, mock_module):
        """Test unsetting an invalid option"""
        mock_framework.session.current_module = mock_module
        mock_module.set_option.return_value = False

        result = command_handler.cmd_unset(['INVALID'])
        assert result is True
        command_handler.display.print_error.assert_called()


class TestRunCommand:
    """Tests for run command"""

    def test_run_no_module_loaded(self, command_handler, mock_framework):
        """Test run command when no module loaded"""
        mock_framework.session.current_module = None

        result = command_handler.cmd_run([])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_run_with_module(self, command_handler, mock_framework, mock_module):
        """Test run command when module is loaded"""
        mock_framework.session.current_module = mock_module
        mock_framework.run_module.return_value = {'total_return': 15.5}

        result = command_handler.cmd_run([])
        assert result is True
        mock_framework.run_module.assert_called_with(mock_module)
        command_handler.display.print_results.assert_called()


class TestSearchCommand:
    """Tests for search command"""

    def test_search_no_args(self, command_handler):
        """Test search command with no arguments"""
        result = command_handler.cmd_search([])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_search_finds_modules(self, command_handler, mock_framework):
        """Test search finds matching modules"""
        mock_framework.search_modules.return_value = [
            {'path': 'strategies/sma_crossover', 'name': 'SMA Crossover'}
        ]

        result = command_handler.cmd_search(['sma'])
        assert result is True
        mock_framework.search_modules.assert_called_with('sma')
        command_handler.display.print_modules.assert_called()

    def test_search_no_results(self, command_handler, mock_framework):
        """Test search with no matching results"""
        mock_framework.search_modules.return_value = []

        result = command_handler.cmd_search(['nonexistent'])
        assert result is True
        command_handler.display.print_warning.assert_called()

    def test_search_multiple_words(self, command_handler, mock_framework):
        """Test search with multiple words"""
        mock_framework.search_modules.return_value = []

        result = command_handler.cmd_search(['sma', 'crossover'])
        mock_framework.search_modules.assert_called_with('sma crossover')


class TestQuoteCommand:
    """Tests for quote command"""

    def test_quote_no_args(self, command_handler):
        """Test quote command with no arguments"""
        result = command_handler.cmd_quote([])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_quote_valid_symbol(self, command_handler, mock_framework):
        """Test quote command with valid symbol"""
        with patch('quantsploit.utils.data_fetcher.DataFetcher') as MockFetcher:
            mock_fetcher = Mock()
            mock_fetcher.get_realtime_quote.return_value = {
                'symbol': 'AAPL',
                'price': 150.0,
                'change': 2.5
            }
            MockFetcher.return_value = mock_fetcher

            result = command_handler.cmd_quote(['aapl'])
            assert result is True
            mock_fetcher.get_realtime_quote.assert_called_with('AAPL')

    def test_quote_failed_fetch(self, command_handler, mock_framework):
        """Test quote command when fetch fails"""
        with patch('quantsploit.utils.data_fetcher.DataFetcher') as MockFetcher:
            mock_fetcher = Mock()
            mock_fetcher.get_realtime_quote.return_value = None
            MockFetcher.return_value = mock_fetcher

            result = command_handler.cmd_quote(['INVALID'])
            assert result is True
            command_handler.display.print_error.assert_called()

    def test_quote_converts_to_uppercase(self, command_handler, mock_framework):
        """Test quote converts symbol to uppercase"""
        with patch('quantsploit.utils.data_fetcher.DataFetcher') as MockFetcher:
            mock_fetcher = Mock()
            mock_fetcher.get_realtime_quote.return_value = {'price': 100}
            MockFetcher.return_value = mock_fetcher

            command_handler.cmd_quote(['msft'])
            mock_fetcher.get_realtime_quote.assert_called_with('MSFT')


class TestWatchlistCommand:
    """Tests for watchlist command"""

    def test_watchlist_no_args_shows_list(self, command_handler):
        """Test watchlist with no args shows list"""
        with patch.object(command_handler, 'cmd_show', return_value=True) as mock_show:
            result = command_handler.cmd_watchlist([])
            mock_show.assert_called_with(['watchlist'])

    def test_watchlist_add(self, command_handler, mock_framework):
        """Test adding to watchlist"""
        mock_framework.database.add_to_watchlist.return_value = True

        result = command_handler.cmd_watchlist(['add', 'NVDA'])
        assert result is True
        mock_framework.database.add_to_watchlist.assert_called_with('NVDA', '')
        command_handler.display.print_success.assert_called()

    def test_watchlist_add_with_notes(self, command_handler, mock_framework):
        """Test adding to watchlist with notes"""
        mock_framework.database.add_to_watchlist.return_value = True

        result = command_handler.cmd_watchlist(['add', 'NVDA', 'AI', 'leader'])
        mock_framework.database.add_to_watchlist.assert_called_with('NVDA', 'AI leader')

    def test_watchlist_add_duplicate(self, command_handler, mock_framework):
        """Test adding duplicate to watchlist"""
        mock_framework.database.add_to_watchlist.return_value = False

        result = command_handler.cmd_watchlist(['add', 'AAPL'])
        assert result is True
        command_handler.display.print_warning.assert_called()

    def test_watchlist_remove(self, command_handler, mock_framework):
        """Test removing from watchlist"""
        result = command_handler.cmd_watchlist(['remove', 'AAPL'])
        assert result is True
        mock_framework.database.remove_from_watchlist.assert_called_with('AAPL')
        command_handler.display.print_success.assert_called()

    def test_watchlist_show(self, command_handler):
        """Test watchlist show subcommand"""
        with patch.object(command_handler, 'cmd_show', return_value=True) as mock_show:
            result = command_handler.cmd_watchlist(['show'])
            mock_show.assert_called_with(['watchlist'])

    def test_watchlist_invalid_action(self, command_handler, mock_framework):
        """Test watchlist with invalid action"""
        result = command_handler.cmd_watchlist(['invalid'])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_watchlist_add_no_symbol(self, command_handler, mock_framework):
        """Test watchlist add without symbol"""
        result = command_handler.cmd_watchlist(['add'])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_watchlist_converts_to_uppercase(self, command_handler, mock_framework):
        """Test watchlist converts symbol to uppercase"""
        mock_framework.database.add_to_watchlist.return_value = True
        command_handler.cmd_watchlist(['add', 'nvda'])
        mock_framework.database.add_to_watchlist.assert_called_with('NVDA', '')


class TestHistoryCommand:
    """Tests for history command"""

    def test_history_with_commands(self, command_handler, mock_framework):
        """Test history shows previous commands"""
        mock_framework.session.command_history = [
            {'command': 'help', 'timestamp': datetime.now()},
            {'command': 'show modules', 'timestamp': datetime.now()}
        ]

        result = command_handler.cmd_history([])
        assert result is True

    def test_history_empty(self, command_handler, mock_framework):
        """Test history when no commands executed"""
        mock_framework.session.command_history = []

        result = command_handler.cmd_history([])
        assert result is True
        command_handler.display.print_info.assert_called()


class TestSessionsCommand:
    """Tests for sessions command"""

    def test_sessions_shows_info(self, command_handler, mock_framework):
        """Test sessions command shows session info"""
        result = command_handler.cmd_sessions([])
        assert result is True
        mock_framework.session.export_session.assert_called()


class TestExitCommand:
    """Tests for exit command"""

    def test_exit_returns_false(self, command_handler):
        """Test exit command returns False to stop loop"""
        result = command_handler.cmd_exit([])
        assert result is False


class TestWebserverCommand:
    """Tests for webserver command"""

    def test_webserver_no_args(self, command_handler):
        """Test webserver command with no arguments"""
        result = command_handler.cmd_webserver([])
        assert result is True
        command_handler.display.print_error.assert_called()

    def test_webserver_start(self, command_handler):
        """Test webserver start command"""
        with patch('quantsploit.modules.webserver.webserver_manager.WebserverManager') as MockManager:
            mock_manager = Mock()
            MockManager.return_value = mock_manager

            result = command_handler.cmd_webserver(['start'])
            assert result is True
            mock_manager.start.assert_called()

    def test_webserver_stop(self, command_handler):
        """Test webserver stop command"""
        with patch('quantsploit.modules.webserver.webserver_manager.WebserverManager') as MockManager:
            mock_manager = Mock()
            MockManager.return_value = mock_manager

            result = command_handler.cmd_webserver(['stop'])
            assert result is True
            mock_manager.stop.assert_called()

    def test_webserver_status(self, command_handler):
        """Test webserver status command"""
        with patch('quantsploit.modules.webserver.webserver_manager.WebserverManager') as MockManager:
            mock_manager = Mock()
            MockManager.return_value = mock_manager

            result = command_handler.cmd_webserver(['status'])
            assert result is True
            mock_manager.status.assert_called()

    def test_webserver_with_port(self, command_handler):
        """Test webserver with port option"""
        with patch('quantsploit.modules.webserver.webserver_manager.WebserverManager') as MockManager:
            mock_manager = Mock()
            MockManager.return_value = mock_manager

            result = command_handler.cmd_webserver(['start', '--port', '8080'])
            assert result is True
            mock_manager.start.assert_called_with('8080', None)

    def test_webserver_invalid_action(self, command_handler):
        """Test webserver with invalid action"""
        with patch('quantsploit.modules.webserver.webserver_manager.WebserverManager') as MockManager:
            mock_manager = Mock()
            MockManager.return_value = mock_manager

            result = command_handler.cmd_webserver(['invalid_action'])
            assert result is True
            command_handler.display.print_error.assert_called()


class TestAnalyzeCommand:
    """Tests for analyze command (if implemented)"""

    def test_analyze_exists(self, command_handler):
        """Test analyze command is registered"""
        assert 'analyze' in command_handler.commands


class TestCompareCommand:
    """Tests for compare command (if implemented)"""

    def test_compare_exists(self, command_handler):
        """Test compare command is registered"""
        assert 'compare' in command_handler.commands


class TestFilterCommand:
    """Tests for filter command (if implemented)"""

    def test_filter_exists(self, command_handler):
        """Test filter command is registered"""
        assert 'filter' in command_handler.commands


class TestInputValidation:
    """Tests for input validation and edge cases"""

    def test_command_parsing_preserves_quotes(self, command_handler, mock_framework, mock_module):
        """Test command parsing handles quoted strings"""
        mock_framework.session.current_module = mock_module
        mock_module.set_option.return_value = True

        # Execute with quoted value
        command_handler.execute('set DESCRIPTION "my description"')
        mock_module.set_option.assert_called_with('DESCRIPTION', 'my description')

    def test_case_insensitive_commands(self, command_handler, mock_framework):
        """Test commands are case insensitive"""
        # HELP should work (converted to lowercase internally)
        result = command_handler.execute('HELP')
        mock_framework.session.add_command.assert_called_with('HELP')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
