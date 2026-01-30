"""
Unit tests for the core BaseModule class

Tests cover:
- Module initialization and option management
- Option setting and retrieval
- Symbol parsing functionality
- Option validation
- Module info display
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.core.module import BaseModule, ModuleMetadata


class ConcreteModule(BaseModule):
    """Concrete implementation of BaseModule for testing"""

    @property
    def name(self) -> str:
        return "Test Module"

    @property
    def description(self) -> str:
        return "A test module for unit testing"

    @property
    def author(self) -> str:
        return "Test Author"

    @property
    def category(self) -> str:
        return "test"

    def run(self) -> Dict[str, Any]:
        return {"status": "completed", "data": self.get_option("SYMBOL")}


class MinimalModule(BaseModule):
    """Minimal implementation with only required properties"""

    @property
    def name(self) -> str:
        return "Minimal"

    @property
    def description(self) -> str:
        return "Minimal module"

    @property
    def author(self) -> str:
        return "Author"

    def run(self) -> Dict[str, Any]:
        return {}


class TestBaseModuleInitialization:
    """Tests for BaseModule initialization"""

    @pytest.fixture
    def mock_framework(self):
        """Create a mock framework"""
        framework = Mock()
        framework.log = Mock()
        return framework

    def test_init_with_framework(self, mock_framework):
        """Test module initialization with framework"""
        module = ConcreteModule(mock_framework)

        assert module.framework == mock_framework
        assert isinstance(module.options, dict)
        assert isinstance(module.results, dict)

    def test_init_default_options(self, mock_framework):
        """Test that default options are initialized"""
        module = ConcreteModule(mock_framework)

        assert "SYMBOL" in module.options
        assert "PERIOD" in module.options
        assert "INTERVAL" in module.options

    def test_default_option_values(self, mock_framework):
        """Test default option values"""
        module = ConcreteModule(mock_framework)

        assert module.options["SYMBOL"]["value"] is None
        assert module.options["SYMBOL"]["required"] is True
        assert module.options["PERIOD"]["value"] == "1y"
        assert module.options["INTERVAL"]["value"] == "1d"


class TestModuleProperties:
    """Tests for module properties"""

    @pytest.fixture
    def module(self):
        framework = Mock()
        return ConcreteModule(framework)

    def test_name_property(self, module):
        """Test name property"""
        assert module.name == "Test Module"

    def test_description_property(self, module):
        """Test description property"""
        assert module.description == "A test module for unit testing"

    def test_author_property(self, module):
        """Test author property"""
        assert module.author == "Test Author"

    def test_category_property(self, module):
        """Test category property"""
        assert module.category == "test"

    def test_default_category(self):
        """Test default category for minimal module"""
        framework = Mock()
        module = MinimalModule(framework)
        assert module.category == "general"

    def test_required_options_default(self, module):
        """Test default required_options is empty list"""
        assert module.required_options == []


class TestOptionManagement:
    """Tests for option setting and getting"""

    @pytest.fixture
    def module(self):
        framework = Mock()
        return ConcreteModule(framework)

    def test_set_option_valid(self, module):
        """Test setting a valid option"""
        result = module.set_option("SYMBOL", "AAPL")

        assert result is True
        assert module.options["SYMBOL"]["value"] == "AAPL"

    def test_set_option_lowercase_key(self, module):
        """Test setting option with lowercase key"""
        result = module.set_option("symbol", "MSFT")

        assert result is True
        assert module.options["SYMBOL"]["value"] == "MSFT"

    def test_set_option_invalid_key(self, module):
        """Test setting an invalid option key"""
        result = module.set_option("NONEXISTENT", "value")

        assert result is False

    def test_get_option_valid(self, module):
        """Test getting a valid option"""
        module.set_option("SYMBOL", "GOOGL")
        value = module.get_option("SYMBOL")

        assert value == "GOOGL"

    def test_get_option_lowercase_key(self, module):
        """Test getting option with lowercase key"""
        module.set_option("SYMBOL", "TSLA")
        value = module.get_option("symbol")

        assert value == "TSLA"

    def test_get_option_invalid_key(self, module):
        """Test getting an invalid option key"""
        value = module.get_option("NONEXISTENT")

        assert value is None

    def test_get_option_default_unset(self, module):
        """Test getting unset required option returns None"""
        value = module.get_option("SYMBOL")

        assert value is None


class TestSymbolParsing:
    """Tests for parse_symbols functionality"""

    @pytest.fixture
    def module(self):
        framework = Mock()
        return ConcreteModule(framework)

    def test_parse_single_symbol(self, module):
        """Test parsing a single symbol"""
        symbols = module.parse_symbols("AAPL")

        assert symbols == ["AAPL"]

    def test_parse_comma_separated(self, module):
        """Test parsing comma-separated symbols"""
        symbols = module.parse_symbols("AAPL,MSFT,GOOGL")

        assert symbols == ["AAPL", "MSFT", "GOOGL"]

    def test_parse_with_spaces(self, module):
        """Test parsing symbols with spaces around commas"""
        symbols = module.parse_symbols("AAPL, MSFT, GOOGL")

        assert symbols == ["AAPL", "MSFT", "GOOGL"]

    def test_parse_lowercase_converts_to_upper(self, module):
        """Test that lowercase symbols are converted to uppercase"""
        symbols = module.parse_symbols("aapl, msft")

        assert symbols == ["AAPL", "MSFT"]

    def test_parse_mixed_case(self, module):
        """Test parsing mixed case symbols"""
        symbols = module.parse_symbols("AaPl, mSfT, GOOGL")

        assert symbols == ["AAPL", "MSFT", "GOOGL"]

    def test_parse_empty_string(self, module):
        """Test parsing empty string"""
        symbols = module.parse_symbols("")

        assert symbols == []

    def test_parse_none(self, module):
        """Test parsing None falls back to options"""
        module.set_option("SYMBOL", "NVDA")
        symbols = module.parse_symbols(None)

        assert symbols == ["NVDA"]

    def test_parse_from_symbols_option(self, module):
        """Test parsing from SYMBOLS option when input is None"""
        module.options["SYMBOLS"] = {"value": "AMD,INTC", "required": False, "description": "Multiple symbols"}
        symbols = module.parse_symbols(None)

        assert symbols == ["AMD", "INTC"]

    def test_parse_list_input(self, module):
        """Test parsing when input is already a list"""
        symbols = module.parse_symbols(["aapl", "msft", "googl"])

        assert symbols == ["AAPL", "MSFT", "GOOGL"]

    def test_parse_removes_empty_entries(self, module):
        """Test that empty entries are removed"""
        symbols = module.parse_symbols("AAPL,,MSFT, ,GOOGL")

        assert symbols == ["AAPL", "MSFT", "GOOGL"]

    def test_parse_non_string_input(self, module):
        """Test parsing non-string input"""
        symbols = module.parse_symbols(12345)

        assert symbols == ["12345"]

    def test_parse_whitespace_trimming(self, module):
        """Test that whitespace is properly trimmed"""
        symbols = module.parse_symbols("  AAPL  ,  MSFT  ")

        assert symbols == ["AAPL", "MSFT"]


class TestOptionValidation:
    """Tests for option validation"""

    @pytest.fixture
    def module(self):
        framework = Mock()
        return ConcreteModule(framework)

    def test_validate_options_missing_required(self, module):
        """Test validation fails when required option is missing"""
        valid, msg = module.validate_options()

        assert valid is False
        assert "SYMBOL" in msg

    def test_validate_options_all_set(self, module):
        """Test validation passes when all required options are set"""
        module.set_option("SYMBOL", "AAPL")
        valid, msg = module.validate_options()

        assert valid is True
        assert msg == "OK"

    def test_validate_with_optional_unset(self, module):
        """Test validation passes with optional options unset"""
        module.set_option("SYMBOL", "AAPL")
        # PERIOD and INTERVAL are optional
        valid, msg = module.validate_options()

        assert valid is True


class TestModuleInfoDisplay:
    """Tests for module info and options display"""

    @pytest.fixture
    def module(self):
        framework = Mock()
        module = ConcreteModule(framework)
        module.set_option("SYMBOL", "AAPL")
        return module

    def test_show_options(self, module):
        """Test show_options returns options dict"""
        options = module.show_options()

        assert isinstance(options, dict)
        assert "SYMBOL" in options
        assert options["SYMBOL"]["value"] == "AAPL"

    def test_show_info(self, module):
        """Test show_info returns module information"""
        info = module.show_info()

        assert info["name"] == "Test Module"
        assert info["description"] == "A test module for unit testing"
        assert info["author"] == "Test Author"
        assert info["category"] == "test"
        assert "options" in info


class TestModuleExecution:
    """Tests for module execution"""

    @pytest.fixture
    def module(self):
        framework = Mock()
        return ConcreteModule(framework)

    def test_run_returns_results(self, module):
        """Test run method returns results"""
        module.set_option("SYMBOL", "AAPL")
        results = module.run()

        assert results["status"] == "completed"
        assert results["data"] == "AAPL"

    def test_cleanup(self, module):
        """Test cleanup method exists and can be called"""
        # Should not raise any exceptions
        module.cleanup()


class TestModuleLogging:
    """Tests for module logging"""

    def test_log_with_framework(self):
        """Test logging through framework"""
        framework = Mock()
        module = ConcreteModule(framework)

        module.log("Test message", "info")

        framework.log.assert_called_once()
        call_args = framework.log.call_args[0]
        assert "Test Module" in call_args[0]
        assert "Test message" in call_args[0]

    def test_log_without_framework(self):
        """Test logging with None framework doesn't crash"""
        module = ConcreteModule(None)
        # Should not raise an exception
        module.log("Test message")


class TestModuleMetadata:
    """Tests for ModuleMetadata class"""

    def test_metadata_creation(self):
        """Test creating module metadata"""
        metadata = ModuleMetadata(
            path="analysis/stock_analyzer",
            name="Stock Analyzer",
            category="analysis",
            description="Analyzes stock data"
        )

        assert metadata.path == "analysis/stock_analyzer"
        assert metadata.name == "Stock Analyzer"
        assert metadata.category == "analysis"
        assert metadata.description == "Analyzes stock data"
        assert metadata.loaded is False
        assert metadata.instance is None

    def test_metadata_loaded_flag(self):
        """Test metadata loaded flag can be set"""
        metadata = ModuleMetadata(
            path="test/module",
            name="Test",
            category="test",
            description="Test module"
        )

        metadata.loaded = True
        assert metadata.loaded is True

    def test_metadata_instance_assignment(self):
        """Test metadata instance can be assigned"""
        metadata = ModuleMetadata(
            path="test/module",
            name="Test",
            category="test",
            description="Test module"
        )

        mock_instance = Mock()
        metadata.instance = mock_instance

        assert metadata.instance == mock_instance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
