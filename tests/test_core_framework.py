"""
Unit tests for the Framework class

Tests cover:
- Framework initialization
- Configuration loading
- Module discovery and registration
- Module loading and usage
- Module search and listing
- Module execution
- Session and database access
- Framework shutdown
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.core.framework import Framework
from quantsploit.core.module import BaseModule, ModuleMetadata
from quantsploit.core.session import Session
from quantsploit.core.database import Database


class TestFrameworkInitialization:
    """Tests for Framework initialization"""

    def test_init_creates_session(self):
        """Test that initialization creates a session"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({"database": {"path": os.path.join(tmpdir, "test.db")}}, f)

            framework = Framework(config_path)

            assert isinstance(framework.session, Session)
            framework.shutdown()

    def test_init_creates_database(self):
        """Test that initialization creates a database"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({"database": {"path": os.path.join(tmpdir, "test.db")}}, f)

            framework = Framework(config_path)

            assert isinstance(framework.database, Database)
            framework.shutdown()

    def test_init_empty_modules(self):
        """Test that initialization starts with empty modules"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({"database": {"path": os.path.join(tmpdir, "test.db")}}, f)

            framework = Framework(config_path)

            assert framework.modules == {}
            framework.shutdown()

    def test_init_empty_log_messages(self):
        """Test that initialization starts with empty log messages"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({"database": {"path": os.path.join(tmpdir, "test.db")}}, f)

            framework = Framework(config_path)

            assert framework.log_messages == []
            framework.shutdown()


class TestConfigLoading:
    """Tests for configuration loading"""

    def test_load_valid_config(self):
        """Test loading a valid configuration file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            config_data = {
                "database": {"path": os.path.join(tmpdir, "test.db")},
                "api_key": "test_key",
                "settings": {"verbose": True}
            }
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)

            framework = Framework(config_path)

            assert framework.config["api_key"] == "test_key"
            assert framework.config["settings"]["verbose"] is True
            framework.shutdown()

    def test_load_nonexistent_config(self):
        """Test loading a non-existent config returns empty dict"""
        with tempfile.TemporaryDirectory() as tmpdir:
            framework = Framework(os.path.join(tmpdir, "nonexistent.yaml"))

            assert framework.config == {}
            framework.shutdown()

    def test_database_path_from_config(self):
        """Test database path is read from config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "custom.db")
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({"database": {"path": db_path}}, f)

            framework = Framework(config_path)

            assert framework.database.db_path == db_path
            framework.shutdown()


class TestLogging:
    """Tests for logging functionality"""

    @pytest.fixture
    def framework(self):
        """Create a framework with temporary config"""
        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump({"database": {"path": os.path.join(tmpdir, "test.db")}}, f)

        framework = Framework(config_path)
        yield framework
        framework.shutdown()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_log_message(self, framework):
        """Test logging a message"""
        framework.log("Test message", "info")

        assert len(framework.log_messages) == 1
        assert framework.log_messages[0]["message"] == "Test message"
        assert framework.log_messages[0]["level"] == "info"

    def test_log_multiple_messages(self, framework):
        """Test logging multiple messages"""
        framework.log("Message 1", "info")
        framework.log("Message 2", "warning")
        framework.log("Message 3", "error")

        assert len(framework.log_messages) == 3
        assert framework.log_messages[1]["level"] == "warning"

    def test_log_includes_timestamp(self, framework):
        """Test that log messages include timestamp"""
        framework.log("Test", "info")

        assert "timestamp" in framework.log_messages[0]


class TestModuleManagement:
    """Tests for module management"""

    @pytest.fixture
    def framework(self):
        """Create a framework with temporary config"""
        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump({"database": {"path": os.path.join(tmpdir, "test.db")}}, f)

        framework = Framework(config_path)
        yield framework
        framework.shutdown()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_module_nonexistent(self, framework):
        """Test getting a non-existent module"""
        result = framework.get_module("nonexistent/module")

        assert result is None

    def test_get_module_exists(self, framework):
        """Test getting an existing module"""
        # Manually add a module for testing
        mock_class = Mock()
        metadata = ModuleMetadata(
            path="test/module",
            name="Test Module",
            category="test",
            description="Test"
        )
        metadata.instance = mock_class
        framework.modules["test/module"] = metadata

        result = framework.get_module("test/module")

        assert result == mock_class

    def test_use_module_nonexistent(self, framework):
        """Test using a non-existent module"""
        result = framework.use_module("nonexistent/module")

        assert result is None

    def test_use_module_exists(self, framework):
        """Test using an existing module"""
        # Create a mock module class
        mock_instance = Mock()
        mock_instance.name = "Test Module"
        mock_class = Mock(return_value=mock_instance)

        metadata = ModuleMetadata(
            path="test/module",
            name="Test Module",
            category="test",
            description="Test"
        )
        metadata.instance = mock_class
        framework.modules["test/module"] = metadata

        result = framework.use_module("test/module")

        assert result == mock_instance
        assert framework.session.current_module == mock_instance

    def test_list_modules_empty(self, framework):
        """Test listing modules when none are loaded"""
        modules = framework.list_modules()

        assert modules == []

    def test_list_modules_all(self, framework):
        """Test listing all modules"""
        for i in range(3):
            metadata = ModuleMetadata(
                path=f"test/module{i}",
                name=f"Module {i}",
                category="test",
                description=f"Description {i}"
            )
            framework.modules[f"test/module{i}"] = metadata

        modules = framework.list_modules()

        assert len(modules) == 3

    def test_list_modules_by_category(self, framework):
        """Test listing modules filtered by category"""
        categories = ["analysis", "strategy", "analysis"]
        for i, cat in enumerate(categories):
            metadata = ModuleMetadata(
                path=f"test/module{i}",
                name=f"Module {i}",
                category=cat,
                description=f"Description {i}"
            )
            framework.modules[f"test/module{i}"] = metadata

        modules = framework.list_modules(category="analysis")

        assert len(modules) == 2
        assert all(m.category == "analysis" for m in modules)


class TestModuleSearch:
    """Tests for module search functionality"""

    @pytest.fixture
    def framework(self):
        """Create a framework with test modules"""
        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump({"database": {"path": os.path.join(tmpdir, "test.db")}}, f)

        framework = Framework(config_path)

        # Add test modules
        test_modules = [
            ("analysis/stock_analyzer", "Stock Analyzer", "analysis", "Analyzes stock data"),
            ("analysis/pattern", "Pattern Recognition", "analysis", "Detects chart patterns"),
            ("strategy/momentum", "Momentum Strategy", "strategy", "Momentum trading signals"),
        ]

        for path, name, cat, desc in test_modules:
            metadata = ModuleMetadata(path, name, cat, desc)
            framework.modules[path] = metadata

        yield framework
        framework.shutdown()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_search_by_name(self, framework):
        """Test searching modules by name"""
        results = framework.search_modules("stock")

        assert len(results) == 1
        assert results[0].name == "Stock Analyzer"

    def test_search_by_description(self, framework):
        """Test searching modules by description"""
        results = framework.search_modules("trading")

        assert len(results) == 1
        assert results[0].name == "Momentum Strategy"

    def test_search_by_path(self, framework):
        """Test searching modules by path"""
        results = framework.search_modules("analysis")

        assert len(results) == 2

    def test_search_case_insensitive(self, framework):
        """Test that search is case insensitive"""
        results = framework.search_modules("STOCK")

        assert len(results) == 1

    def test_search_no_results(self, framework):
        """Test search with no matching results"""
        results = framework.search_modules("nonexistent")

        assert results == []


class TestModuleExecution:
    """Tests for module execution"""

    @pytest.fixture
    def framework(self):
        """Create a framework with temporary config"""
        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump({"database": {"path": os.path.join(tmpdir, "test.db")}}, f)

        framework = Framework(config_path)
        yield framework
        framework.shutdown()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_module_validation_failure(self, framework):
        """Test running a module that fails validation"""
        mock_module = Mock()
        mock_module.validate_options.return_value = (False, "Missing SYMBOL")

        result = framework.run_module(mock_module)

        assert result["success"] is False
        assert "Missing SYMBOL" in result["error"]

    def test_run_module_success(self, framework):
        """Test running a module successfully"""
        mock_module = Mock()
        mock_module.name = "TestModule"
        mock_module.validate_options.return_value = (True, "OK")
        mock_module.run.return_value = {"data": "test_result"}
        mock_module.get_option.return_value = "AAPL"
        mock_module.options = {"SYMBOL": {"value": "AAPL"}}

        result = framework.run_module(mock_module)

        assert result["success"] is True
        assert result["data"] == "test_result"

    def test_run_module_stores_in_session(self, framework):
        """Test that running a module stores results in session"""
        mock_module = Mock()
        mock_module.name = "TestModule"
        mock_module.validate_options.return_value = (True, "OK")
        mock_module.run.return_value = {"data": "result"}
        mock_module.get_option.return_value = "AAPL"
        mock_module.options = {"SYMBOL": {"value": "AAPL"}}

        framework.run_module(mock_module)

        results = framework.session.get_results("TestModule")
        assert len(results) == 1

    def test_run_module_exception_handling(self, framework):
        """Test that module exceptions are handled"""
        mock_module = Mock()
        mock_module.validate_options.return_value = (True, "OK")
        mock_module.run.side_effect = Exception("Test error")

        result = framework.run_module(mock_module)

        assert result["success"] is False
        assert "Test error" in result["error"]


class TestAccessors:
    """Tests for accessor methods"""

    @pytest.fixture
    def framework(self):
        """Create a framework with temporary config"""
        tmpdir = tempfile.mkdtemp()
        config_path = os.path.join(tmpdir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump({"database": {"path": os.path.join(tmpdir, "test.db")}}, f)

        framework = Framework(config_path)
        yield framework
        framework.shutdown()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_session(self, framework):
        """Test getting the session"""
        session = framework.get_session()

        assert session == framework.session
        assert isinstance(session, Session)

    def test_get_database(self, framework):
        """Test getting the database"""
        database = framework.get_database()

        assert database == framework.database
        assert isinstance(database, Database)


class TestFrameworkShutdown:
    """Tests for framework shutdown"""

    def test_shutdown_closes_database(self):
        """Test that shutdown closes the database"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({"database": {"path": os.path.join(tmpdir, "test.db")}}, f)

            framework = Framework(config_path)

            # Mock the database close method
            framework.database.close = Mock()

            framework.shutdown()

            framework.database.close.assert_called_once()

    def test_shutdown_logs_message(self):
        """Test that shutdown logs a message"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({"database": {"path": os.path.join(tmpdir, "test.db")}}, f)

            framework = Framework(config_path)
            framework.shutdown()

            # Check that shutdown message was logged
            shutdown_messages = [m for m in framework.log_messages if "shutdown" in m["message"].lower()]
            assert len(shutdown_messages) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
