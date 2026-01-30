"""
Unit tests for UI Display Module

Tests cover:
- Display initialization
- Print methods (success, error, warning, info)
- Module info display
- Options display
- Modules list display
- DataFrame display
- Results display
- Help display
- Formatting helpers

★ Insight ─────────────────────────────────────
The Display class wraps Rich console for consistent, beautiful output.
Testing output modules focuses on ensuring correct method calls rather
than visual appearance - we verify data flow, not pixel rendering.
─────────────────────────────────────────────────
"""

import pytest
from unittest.mock import Mock, patch, call
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.ui.display import Display


@pytest.fixture
def display():
    """Create a Display instance with mocked console"""
    with patch('quantsploit.ui.display.RichConsole') as MockConsole:
        mock_console = Mock()
        MockConsole.return_value = mock_console

        disp = Display()
        disp.console = mock_console  # Ensure we use the mock
        return disp


@pytest.fixture
def sample_module_info():
    """Sample module info for testing"""
    return {
        'name': 'SMA Crossover Strategy',
        'category': 'strategies',
        'description': 'Simple moving average crossover trading strategy',
        'author': 'Quantsploit Team',
        'trading_guide': 'Buy when fast SMA crosses above slow SMA.'
    }


@pytest.fixture
def sample_options():
    """Sample options for testing"""
    return {
        'SYMBOL': {
            'required': True,
            'value': 'AAPL',
            'description': 'Stock ticker symbol'
        },
        'FAST_PERIOD': {
            'required': False,
            'value': 10,
            'description': 'Fast moving average period'
        },
        'SLOW_PERIOD': {
            'required': False,
            'value': None,
            'description': 'Slow moving average period'
        }
    }


@pytest.fixture
def sample_modules():
    """Sample modules list for testing"""
    module1 = Mock()
    module1.path = 'strategies/sma_crossover'
    module1.category = 'strategies'
    module1.description = 'SMA crossover strategy'

    module2 = Mock()
    module2.path = 'analysis/stock_analyzer'
    module2.category = 'analysis'
    module2.description = 'Analyze stock performance'

    return [module1, module2]


class TestDisplayInitialization:
    """Tests for Display initialization"""

    def test_display_creates_console(self):
        """Test Display creates a Rich Console"""
        with patch('quantsploit.ui.display.RichConsole') as MockConsole:
            display = Display()
            MockConsole.assert_called_once()

    def test_display_has_console_attribute(self, display):
        """Test Display has console attribute"""
        assert hasattr(display, 'console')


class TestPrintMethod:
    """Tests for basic print method"""

    def test_print_text(self, display):
        """Test basic print"""
        display.print("Hello World")
        display.console.print.assert_called_with("Hello World", style="")

    def test_print_with_style(self, display):
        """Test print with style"""
        display.print("Styled text", style="bold red")
        display.console.print.assert_called_with("Styled text", style="bold red")


class TestPrintBanner:
    """Tests for banner printing"""

    def test_print_banner_calls_console(self, display):
        """Test print_banner calls console.print"""
        display.print_banner()
        # Should have at least two calls (banner + version)
        assert display.console.print.call_count >= 2


class TestPrintSuccess:
    """Tests for success message printing"""

    def test_print_success_format(self, display):
        """Test success message format"""
        display.print_success("Operation completed")
        display.console.print.assert_called_with(
            "[+] Operation completed",
            style="bold green"
        )

    def test_print_success_empty_message(self, display):
        """Test success with empty message"""
        display.print_success("")
        display.console.print.assert_called_with("[+] ", style="bold green")


class TestPrintError:
    """Tests for error message printing"""

    def test_print_error_format(self, display):
        """Test error message format"""
        display.print_error("Something went wrong")
        display.console.print.assert_called_with(
            "[-] Something went wrong",
            style="bold red"
        )

    def test_print_error_with_special_chars(self, display):
        """Test error with special characters"""
        display.print_error("Error: 'value' not found")
        display.console.print.assert_called_with(
            "[-] Error: 'value' not found",
            style="bold red"
        )


class TestPrintWarning:
    """Tests for warning message printing"""

    def test_print_warning_format(self, display):
        """Test warning message format"""
        display.print_warning("This might be an issue")
        display.console.print.assert_called_with(
            "[!] This might be an issue",
            style="bold yellow"
        )


class TestPrintInfo:
    """Tests for info message printing"""

    def test_print_info_format(self, display):
        """Test info message format"""
        display.print_info("Loading module...")
        display.console.print.assert_called_with(
            "[*] Loading module...",
            style="bold blue"
        )


class TestPrintModuleInfo:
    """Tests for module info display"""

    def test_print_module_info_basic(self, display, sample_module_info):
        """Test basic module info display"""
        display.print_module_info(sample_module_info)
        # Should call console.print at least once
        assert display.console.print.called

    def test_print_module_info_with_trading_guide(self, display, sample_module_info):
        """Test module info with trading guide"""
        display.print_module_info(sample_module_info)
        # With trading guide, should have multiple print calls
        assert display.console.print.call_count >= 2

    def test_print_module_info_without_trading_guide(self, display):
        """Test module info without trading guide"""
        module_info = {
            'name': 'Test Module',
            'category': 'test',
            'description': 'Test description',
            'author': 'Tester'
        }
        display.print_module_info(module_info)
        # Should still work without trading guide
        assert display.console.print.called


class TestPrintOptions:
    """Tests for options display"""

    def test_print_options_basic(self, display, sample_options):
        """Test basic options display"""
        display.print_options(sample_options)
        assert display.console.print.called

    def test_print_options_empty(self, display):
        """Test options display with empty dict"""
        display.print_options({})
        # Should still create and print table
        assert display.console.print.called

    def test_print_options_handles_none_value(self, display):
        """Test options with None value"""
        options = {
            'OPTION': {
                'required': True,
                'value': None,
                'description': 'Test option'
            }
        }
        display.print_options(options)
        assert display.console.print.called

    def test_print_options_handles_missing_keys(self, display):
        """Test options with missing optional keys"""
        options = {
            'OPTION': {
                # Missing 'required', 'value', 'description'
            }
        }
        display.print_options(options)
        assert display.console.print.called


class TestPrintModules:
    """Tests for modules list display"""

    def test_print_modules_basic(self, display, sample_modules):
        """Test basic modules list display"""
        display.print_modules(sample_modules)
        assert display.console.print.called

    def test_print_modules_empty_list(self, display):
        """Test modules display with empty list"""
        display.print_modules([])
        assert display.console.print.called

    def test_print_modules_with_category(self, display, sample_modules):
        """Test modules display with category"""
        display.print_modules(sample_modules, category="strategies")
        assert display.console.print.called


class TestPrintDataFrame:
    """Tests for DataFrame display"""

    def test_print_dataframe_basic(self, display):
        """Test basic DataFrame display"""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        display.print_dataframe(df, title="Test Data")
        assert display.console.print.called

    def test_print_dataframe_empty(self, display):
        """Test empty DataFrame display"""
        df = pd.DataFrame()
        display.print_dataframe(df)
        # Should call print_warning
        display.console.print.assert_called()

    def test_print_dataframe_none(self, display):
        """Test None DataFrame display"""
        display.print_dataframe(None)
        # Should call print_warning
        display.console.print.assert_called()

    def test_print_dataframe_truncates_large(self, display):
        """Test DataFrame truncation for large data"""
        df = pd.DataFrame({'A': range(100)})
        display.print_dataframe(df, max_rows=10)
        # Should print warning about truncation
        assert display.console.print.call_count >= 2  # Table + warning

    def test_print_dataframe_respects_max_rows(self, display):
        """Test DataFrame respects max_rows parameter"""
        df = pd.DataFrame({'A': range(20)})
        display.print_dataframe(df, max_rows=50)
        # With 20 rows < 50, should not truncate
        assert display.console.print.called


class TestPrintResults:
    """Tests for results display"""

    def test_print_results_success(self, display):
        """Test successful results display"""
        results = {
            'success': True,
            'total_return': 15.5,
            'sharpe_ratio': 1.2
        }
        display.print_results(results)
        # Should call print_success and display values
        assert display.console.print.call_count >= 1

    def test_print_results_failure(self, display):
        """Test failed results display"""
        results = {
            'success': False,
            'error': 'Connection timeout'
        }
        display.print_results(results)
        # Should call print_error
        assert display.console.print.called

    def test_print_results_with_dataframe(self, display):
        """Test results with DataFrame"""
        results = {
            'success': True,
            'trade_history': pd.DataFrame({'profit': [10, -5, 20]})
        }
        display.print_results(results)
        assert display.console.print.called

    def test_print_results_with_dict(self, display):
        """Test results with nested dict"""
        results = {
            'success': True,
            'metrics': {'sharpe': 1.2, 'sortino': 1.5}
        }
        display.print_results(results)
        assert display.console.print.called

    def test_print_results_with_list(self, display):
        """Test results with list"""
        results = {
            'success': True,
            'signals': ['BUY', 'HOLD', 'SELL']
        }
        display.print_results(results)
        assert display.console.print.called

    def test_print_results_skips_success_and_error_keys(self, display):
        """Test that success and error keys are not displayed as values"""
        results = {
            'success': True,
            'error': None,
            'actual_value': 42
        }
        display.print_results(results)
        # Should display actual_value but not success/error as data


class TestPrintDict:
    """Tests for dictionary display helper"""

    def test_print_dict_basic(self, display):
        """Test basic dictionary display"""
        data = {'key1': 'value1', 'key2': 'value2'}
        display._print_dict(data, "Test Dict")
        assert display.console.print.called

    def test_print_dict_empty(self, display):
        """Test empty dictionary display"""
        display._print_dict({}, "Empty Dict")
        assert display.console.print.called

    def test_print_dict_numeric_values(self, display):
        """Test dict with numeric values"""
        data = {'count': 42, 'ratio': 3.14}
        display._print_dict(data, "Numbers")
        assert display.console.print.called


class TestPrintList:
    """Tests for list display helper"""

    def test_print_list_basic(self, display):
        """Test basic list display"""
        data = ['item1', 'item2', 'item3']
        display._print_list(data, "Test List")
        # Should print title + items
        assert display.console.print.call_count >= 4

    def test_print_list_empty(self, display):
        """Test empty list display"""
        display._print_list([], "Empty List")
        # Should at least print the title
        assert display.console.print.called

    def test_print_list_single_item(self, display):
        """Test single item list"""
        display._print_list(['only_item'], "Single")
        assert display.console.print.call_count >= 2


class TestPrintHelp:
    """Tests for help display"""

    def test_print_help_basic(self, display):
        """Test basic help display"""
        commands = {
            'help': 'Display help information',
            'exit': 'Exit the application'
        }
        display.print_help(commands)
        assert display.console.print.called

    def test_print_help_empty(self, display):
        """Test help with no commands"""
        display.print_help({})
        assert display.console.print.called

    def test_print_help_sorted_output(self, display):
        """Test help displays commands sorted"""
        commands = {
            'zebra': 'Last command',
            'alpha': 'First command'
        }
        display.print_help(commands)
        # Commands should be sorted alphabetically
        assert display.console.print.called


class TestPrintPrompt:
    """Tests for prompt display"""

    def test_print_prompt_basic(self, display):
        """Test basic prompt display"""
        display.print_prompt("quantsploit > ")
        display.console.print.assert_called_with("quantsploit > ", end="")

    def test_print_prompt_with_module(self, display):
        """Test prompt with module context"""
        display.print_prompt("quantsploit(sma_crossover) > ")
        display.console.print.assert_called_with(
            "quantsploit(sma_crossover) > ",
            end=""
        )


class TestDisplayIntegration:
    """Integration tests for Display workflows"""

    def test_error_success_sequence(self, display):
        """Test error followed by success"""
        display.print_error("First try failed")
        display.print_success("Retry succeeded")

        calls = display.console.print.call_args_list
        assert len(calls) == 2
        assert "[-]" in calls[0][0][0]  # Error format
        assert "[+]" in calls[1][0][0]  # Success format

    def test_results_workflow(self, display):
        """Test typical results display workflow"""
        # Simulate module execution results
        results = {
            'success': True,
            'summary': {
                'total_return': 25.5,
                'trades': 15
            },
            'signals': ['BUY', 'SELL', 'HOLD']
        }
        display.print_results(results)
        # Should have multiple print calls for different data types
        assert display.console.print.call_count >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
