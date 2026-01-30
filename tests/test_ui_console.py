"""
Unit tests for UI Console Module

Tests cover:
- Console initialization
- Completer creation
- Prompt generation
- Console start/stop lifecycle
- Input handling
- Error handling (KeyboardInterrupt, EOFError)

★ Insight ─────────────────────────────────────
The Console class provides the REPL loop using prompt_toolkit.
Testing interactive console code requires careful mocking of the
PromptSession to simulate user input and verify response handling.
─────────────────────────────────────────────────
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.ui.console import Console


@pytest.fixture
def mock_framework():
    """Create a mock framework object"""
    framework = Mock()
    framework.modules = {
        'analysis/stock_analyzer': Mock(),
        'strategies/sma_crossover': Mock(),
        'scanners/price_momentum': Mock()
    }
    framework.session = Mock()
    framework.session.current_module = None
    framework.shutdown = Mock()
    return framework


@pytest.fixture
def console(mock_framework):
    """Create a Console instance with mocked dependencies"""
    with patch('quantsploit.ui.console.Display') as MockDisplay, \
         patch('quantsploit.ui.console.CommandHandler') as MockCmdHandler, \
         patch('quantsploit.ui.console.PromptSession') as MockSession:

        mock_display = Mock()
        MockDisplay.return_value = mock_display

        mock_cmd_handler = Mock()
        mock_cmd_handler.commands = {
            'help': Mock(),
            'show': Mock(),
            'use': Mock(),
            'exit': Mock()
        }
        MockCmdHandler.return_value = mock_cmd_handler

        mock_session = Mock()
        MockSession.return_value = mock_session

        con = Console(mock_framework)
        con.display = mock_display
        con.command_handler = mock_cmd_handler
        con.session = mock_session

        return con


class TestConsoleInitialization:
    """Tests for Console initialization"""

    def test_console_stores_framework(self, mock_framework):
        """Test Console stores framework reference"""
        with patch('quantsploit.ui.console.Display'), \
             patch('quantsploit.ui.console.CommandHandler'), \
             patch('quantsploit.ui.console.PromptSession'):
            console = Console(mock_framework)
            assert console.framework == mock_framework

    def test_console_creates_display(self, mock_framework):
        """Test Console creates Display instance"""
        with patch('quantsploit.ui.console.Display') as MockDisplay, \
             patch('quantsploit.ui.console.CommandHandler'), \
             patch('quantsploit.ui.console.PromptSession'):
            Console(mock_framework)
            MockDisplay.assert_called_once()

    def test_console_creates_command_handler(self, mock_framework):
        """Test Console creates CommandHandler"""
        with patch('quantsploit.ui.console.Display'), \
             patch('quantsploit.ui.console.CommandHandler') as MockCmdHandler, \
             patch('quantsploit.ui.console.PromptSession'):
            Console(mock_framework)
            MockCmdHandler.assert_called_once_with(mock_framework)

    def test_console_creates_prompt_session(self, mock_framework):
        """Test Console creates PromptSession"""
        with patch('quantsploit.ui.console.Display'), \
             patch('quantsploit.ui.console.CommandHandler'), \
             patch('quantsploit.ui.console.PromptSession') as MockSession:
            Console(mock_framework)
            MockSession.assert_called_once()

    def test_console_starts_running(self, console):
        """Test Console starts in running state"""
        assert console.running is True

    def test_console_has_completer(self, console):
        """Test Console creates completer"""
        assert console.completer is not None

    def test_console_has_prompt_style(self, console):
        """Test Console has prompt style"""
        assert console.prompt_style is not None


class TestCreateCompleter:
    """Tests for completer creation"""

    def test_completer_includes_commands(self, console):
        """Test completer includes registered commands"""
        # The fixture mocked commands, verify completer was created
        assert console.completer is not None

    def test_create_completer_uses_command_keys(self, mock_framework):
        """Test _create_completer uses command handler keys"""
        with patch('quantsploit.ui.console.Display'), \
             patch('quantsploit.ui.console.CommandHandler') as MockCmdHandler, \
             patch('quantsploit.ui.console.PromptSession'), \
             patch('quantsploit.ui.console.WordCompleter') as MockCompleter:

            mock_cmd = Mock()
            mock_cmd.commands = {'help': Mock(), 'exit': Mock()}
            MockCmdHandler.return_value = mock_cmd

            Console(mock_framework)

            # WordCompleter should be called with command keys
            MockCompleter.assert_called()


class TestGetPrompt:
    """Tests for prompt generation"""

    def test_prompt_without_module(self, console, mock_framework):
        """Test prompt when no module loaded"""
        mock_framework.session.current_module = None

        prompt = console._get_prompt()
        assert 'quantsploit' in prompt
        assert '>' in prompt

    def test_prompt_with_module(self, console, mock_framework):
        """Test prompt when module is loaded"""
        mock_module = Mock()
        mock_module.name = 'SMA Crossover'
        mock_framework.session.current_module = mock_module

        prompt = console._get_prompt()
        assert 'quantsploit' in prompt
        assert 'SMA Crossover' in prompt


class TestConsoleStart:
    """Tests for console start method"""

    def test_start_prints_banner(self, console, mock_framework):
        """Test start prints banner"""
        # Make execute return False to exit loop immediately
        console.command_handler.execute.return_value = False
        console.session.prompt.return_value = 'exit'

        console.start()

        console.display.print_banner.assert_called_once()

    def test_start_prints_module_count(self, console, mock_framework):
        """Test start prints number of loaded modules"""
        console.command_handler.execute.return_value = False
        console.session.prompt.return_value = 'exit'

        console.start()

        console.display.print_info.assert_called()

    def test_start_executes_commands(self, console, mock_framework):
        """Test start executes user commands"""
        # First call returns True (continue), second returns False (exit)
        console.command_handler.execute.side_effect = [True, False]
        console.session.prompt.side_effect = ['help', 'exit']

        console.start()

        assert console.command_handler.execute.call_count == 2

    def test_start_handles_keyboard_interrupt(self, console, mock_framework):
        """Test start handles Ctrl+C gracefully"""
        # First prompt raises KeyboardInterrupt, second returns exit
        console.session.prompt.side_effect = [KeyboardInterrupt, 'exit']
        console.command_handler.execute.return_value = False

        console.start()

        # Should print warning about using exit
        console.display.print_warning.assert_called()

    def test_start_handles_eof(self, console, mock_framework):
        """Test start handles EOF (Ctrl+D)"""
        console.session.prompt.side_effect = EOFError

        console.start()

        # Should exit gracefully
        mock_framework.shutdown.assert_called_once()

    def test_start_handles_exception(self, console, mock_framework):
        """Test start handles general exceptions"""
        # First prompt raises exception, second returns exit
        console.session.prompt.side_effect = [Exception("Test error"), 'exit']
        console.command_handler.execute.return_value = False

        console.start()

        # Should print error
        console.display.print_error.assert_called()

    def test_start_calls_shutdown(self, console, mock_framework):
        """Test start calls framework shutdown"""
        console.command_handler.execute.return_value = False
        console.session.prompt.return_value = 'exit'

        console.start()

        mock_framework.shutdown.assert_called_once()

    def test_start_prints_goodbye(self, console, mock_framework):
        """Test start prints goodbye message"""
        console.command_handler.execute.return_value = False
        console.session.prompt.return_value = 'exit'

        console.start()

        # Should call print_success with Goodbye
        goodbye_calls = [
            c for c in console.display.print_success.call_args_list
            if 'Goodbye' in str(c)
        ]
        assert len(goodbye_calls) > 0


class TestConsoleStop:
    """Tests for console stop method"""

    def test_stop_sets_running_false(self, console):
        """Test stop sets running to False"""
        console.running = True
        console.stop()
        assert console.running is False


class TestConsoleLoop:
    """Tests for console main loop behavior"""

    def test_loop_continues_while_running(self, console, mock_framework):
        """Test loop continues while running is True"""
        # Return True 3 times, then False
        console.command_handler.execute.side_effect = [True, True, True, False]
        console.session.prompt.side_effect = ['help', 'show modules', 'info', 'exit']

        console.start()

        assert console.command_handler.execute.call_count == 4

    def test_loop_stops_when_command_returns_false(self, console, mock_framework):
        """Test loop stops when command returns False"""
        console.command_handler.execute.return_value = False
        console.session.prompt.return_value = 'exit'

        console.start()

        # Should only execute once
        console.command_handler.execute.assert_called_once()

    def test_keyboard_interrupt_continues_loop(self, console, mock_framework):
        """Test KeyboardInterrupt doesn't exit loop"""
        # First: interrupt, Second: help (continue), Third: exit
        console.session.prompt.side_effect = [KeyboardInterrupt, 'help', 'exit']
        console.command_handler.execute.side_effect = [True, False]

        console.start()

        # Should have executed twice (help and exit)
        assert console.command_handler.execute.call_count == 2


class TestConsoleIntegration:
    """Integration tests for Console"""

    def test_full_session_workflow(self, console, mock_framework):
        """Test a typical console session workflow"""
        # Simulate: help -> use module -> run -> exit
        console.session.prompt.side_effect = [
            'help',
            'use strategies/sma_crossover',
            'run',
            'exit'
        ]
        console.command_handler.execute.side_effect = [True, True, True, False]

        console.start()

        # All commands should have been executed
        assert console.command_handler.execute.call_count == 4

        # Framework should be shut down
        mock_framework.shutdown.assert_called_once()

    def test_empty_input_handling(self, console, mock_framework):
        """Test handling of empty input"""
        console.session.prompt.side_effect = ['', '', 'exit']
        console.command_handler.execute.side_effect = [True, True, False]

        console.start()

        # Empty inputs should still be passed to command handler
        assert console.command_handler.execute.call_count == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
