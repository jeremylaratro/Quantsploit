"""
Unit tests for the Session class

Tests cover:
- Session initialization
- Module loading/unloading
- Variable management
- Results storage and retrieval
- Command history tracking
- Session export
"""

import pytest
from unittest.mock import Mock
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.core.session import Session


class TestSessionInitialization:
    """Tests for Session initialization"""

    def test_init_creates_empty_state(self):
        """Test session initializes with empty state"""
        session = Session()

        assert session.current_module is None
        assert session.module_history == []
        assert session.workspace == {}
        assert session.variables == {}
        assert session.command_history == []

    def test_init_sets_created_at(self):
        """Test session sets creation timestamp"""
        before = datetime.now()
        session = Session()
        after = datetime.now()

        assert before <= session.created_at <= after


class TestModuleManagement:
    """Tests for module loading and unloading"""

    @pytest.fixture
    def session(self):
        return Session()

    @pytest.fixture
    def mock_module(self):
        module = Mock()
        module.name = "TestModule"
        return module

    def test_load_module_sets_current(self, session, mock_module):
        """Test loading a module sets it as current"""
        session.load_module(mock_module)

        assert session.current_module == mock_module

    def test_load_module_adds_previous_to_history(self, session):
        """Test loading new module adds previous to history"""
        module1 = Mock()
        module1.name = "Module1"
        module2 = Mock()
        module2.name = "Module2"

        session.load_module(module1)
        session.load_module(module2)

        assert session.current_module == module2
        assert len(session.module_history) == 1
        assert session.module_history[0]["module"] == "Module1"
        assert "unloaded_at" in session.module_history[0]

    def test_load_multiple_modules(self, session):
        """Test loading multiple modules in sequence"""
        modules = [Mock(name=f"Module{i}") for i in range(3)]
        for m in modules:
            m.name = m._mock_name

        for module in modules:
            session.load_module(module)

        assert session.current_module == modules[-1]
        assert len(session.module_history) == 2

    def test_unload_module(self, session, mock_module):
        """Test unloading the current module"""
        session.load_module(mock_module)
        session.unload_module()

        assert session.current_module is None
        assert len(session.module_history) == 1
        assert session.module_history[0]["module"] == "TestModule"

    def test_unload_when_no_module(self, session):
        """Test unloading when no module is loaded"""
        session.unload_module()

        assert session.current_module is None
        assert len(session.module_history) == 0


class TestVariableManagement:
    """Tests for session variable management"""

    @pytest.fixture
    def session(self):
        return Session()

    def test_set_variable(self, session):
        """Test setting a session variable"""
        session.set_variable("test_key", "test_value")

        assert session.variables["test_key"] == "test_value"

    def test_get_variable_exists(self, session):
        """Test getting an existing variable"""
        session.set_variable("key", "value")
        result = session.get_variable("key")

        assert result == "value"

    def test_get_variable_not_exists(self, session):
        """Test getting a non-existent variable returns None"""
        result = session.get_variable("nonexistent")

        assert result is None

    def test_set_variable_overwrite(self, session):
        """Test overwriting an existing variable"""
        session.set_variable("key", "original")
        session.set_variable("key", "updated")

        assert session.get_variable("key") == "updated"

    def test_set_variable_various_types(self, session):
        """Test setting variables of various types"""
        session.set_variable("string", "test")
        session.set_variable("number", 42)
        session.set_variable("list", [1, 2, 3])
        session.set_variable("dict", {"a": 1})

        assert session.get_variable("string") == "test"
        assert session.get_variable("number") == 42
        assert session.get_variable("list") == [1, 2, 3]
        assert session.get_variable("dict") == {"a": 1}


class TestResultsStorage:
    """Tests for results storage and retrieval"""

    @pytest.fixture
    def session(self):
        return Session()

    def test_store_results_new_module(self, session):
        """Test storing results for a new module"""
        results = {"data": "test_data", "count": 10}
        session.store_results("TestModule", results)

        assert "TestModule" in session.workspace
        assert len(session.workspace["TestModule"]) == 1
        assert session.workspace["TestModule"][0]["results"] == results

    def test_store_results_adds_timestamp(self, session):
        """Test that stored results include timestamp"""
        session.store_results("TestModule", {"data": "test"})

        stored = session.workspace["TestModule"][0]
        assert "timestamp" in stored
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(stored["timestamp"])

    def test_store_results_multiple_same_module(self, session):
        """Test storing multiple results for the same module"""
        session.store_results("TestModule", {"run": 1})
        session.store_results("TestModule", {"run": 2})
        session.store_results("TestModule", {"run": 3})

        assert len(session.workspace["TestModule"]) == 3

    def test_store_results_multiple_modules(self, session):
        """Test storing results for different modules"""
        session.store_results("Module1", {"data": 1})
        session.store_results("Module2", {"data": 2})

        assert "Module1" in session.workspace
        assert "Module2" in session.workspace

    def test_get_results_existing_module(self, session):
        """Test getting results for an existing module"""
        session.store_results("TestModule", {"data": "test"})
        results = session.get_results("TestModule")

        assert len(results) == 1
        assert results[0]["results"]["data"] == "test"

    def test_get_results_nonexistent_module(self, session):
        """Test getting results for a non-existent module"""
        results = session.get_results("NonExistent")

        assert results == []


class TestCommandHistory:
    """Tests for command history tracking"""

    @pytest.fixture
    def session(self):
        return Session()

    def test_add_command(self, session):
        """Test adding a command to history"""
        session.add_command("use analysis/stock_analyzer")

        assert len(session.command_history) == 1
        assert session.command_history[0]["command"] == "use analysis/stock_analyzer"

    def test_add_command_includes_timestamp(self, session):
        """Test that added commands include timestamp"""
        before = datetime.now()
        session.add_command("test command")
        after = datetime.now()

        timestamp = session.command_history[0]["timestamp"]
        assert before <= timestamp <= after

    def test_add_multiple_commands(self, session):
        """Test adding multiple commands preserves order"""
        commands = ["command1", "command2", "command3"]
        for cmd in commands:
            session.add_command(cmd)

        assert len(session.command_history) == 3
        for i, cmd in enumerate(commands):
            assert session.command_history[i]["command"] == cmd


class TestWorkspaceManagement:
    """Tests for workspace management"""

    @pytest.fixture
    def session(self):
        return Session()

    def test_clear_workspace(self, session):
        """Test clearing the workspace"""
        session.store_results("Module1", {"data": 1})
        session.store_results("Module2", {"data": 2})

        session.clear_workspace()

        assert session.workspace == {}

    def test_clear_workspace_empty(self, session):
        """Test clearing an already empty workspace"""
        session.clear_workspace()

        assert session.workspace == {}


class TestSessionExport:
    """Tests for session export functionality"""

    @pytest.fixture
    def session(self):
        return Session()

    def test_export_empty_session(self, session):
        """Test exporting an empty session"""
        export = session.export_session()

        assert "created_at" in export
        assert export["current_module"] is None
        assert export["variables"] == {}
        assert export["workspace"] == {}
        assert export["command_history"] == []

    def test_export_with_module(self, session):
        """Test exporting session with loaded module"""
        module = Mock()
        module.name = "TestModule"
        session.load_module(module)

        export = session.export_session()

        assert export["current_module"] == "TestModule"

    def test_export_with_variables(self, session):
        """Test exporting session with variables"""
        session.set_variable("key1", "value1")
        session.set_variable("key2", 42)

        export = session.export_session()

        assert export["variables"]["key1"] == "value1"
        assert export["variables"]["key2"] == 42

    def test_export_with_workspace(self, session):
        """Test exporting session with workspace data"""
        session.store_results("TestModule", {"data": "test"})

        export = session.export_session()

        assert "TestModule" in export["workspace"]

    def test_export_with_command_history(self, session):
        """Test exporting session with command history"""
        session.add_command("command1")
        session.add_command("command2")

        export = session.export_session()

        assert len(export["command_history"]) == 2
        assert export["command_history"][0]["cmd"] == "command1"
        assert export["command_history"][1]["cmd"] == "command2"
        # Verify timestamps are ISO format strings
        for entry in export["command_history"]:
            assert "ts" in entry
            datetime.fromisoformat(entry["ts"])

    def test_export_created_at_is_iso_format(self, session):
        """Test that created_at is exported as ISO format string"""
        export = session.export_session()

        # Should be parseable as ISO format
        datetime.fromisoformat(export["created_at"])

    def test_export_full_session(self, session):
        """Test exporting a fully populated session"""
        # Set up a complete session
        module = Mock()
        module.name = "ActiveModule"
        session.load_module(module)

        session.set_variable("api_key", "test_key")
        session.store_results("ActiveModule", {"analysis": "complete"})
        session.add_command("use analysis/stock_analyzer")
        session.add_command("set SYMBOL AAPL")
        session.add_command("run")

        export = session.export_session()

        assert export["current_module"] == "ActiveModule"
        assert export["variables"]["api_key"] == "test_key"
        assert "ActiveModule" in export["workspace"]
        assert len(export["command_history"]) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
