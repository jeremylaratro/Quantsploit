"""
Main framework engine for Quantsploit
"""

import os
import importlib
import inspect
from typing import Dict, List, Optional, Type
from pathlib import Path
import yaml

from .module import BaseModule, ModuleMetadata
from .session import Session
from .database import Database


class Framework:
    """
    Core framework that manages modules, sessions, and data
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.modules = {}  # path -> ModuleMetadata
        self.session = Session()
        self.database = None
        self.config = self._load_config(config_path)
        self.log_messages = []
        self._init_database()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _init_database(self):
        """Initialize database connection"""
        db_path = self.config.get("database", {}).get("path", "./quantsploit.db")
        self.database = Database(db_path)

    def log(self, message: str, level: str = "info"):
        """Log a message"""
        self.log_messages.append({
            "level": level,
            "message": message,
            "timestamp": str(Session().created_at)
        })

    def discover_modules(self, base_path: str = None):
        """
        Discover and register all available modules.
        Similar to Metasploit's module loading.
        """
        if base_path is None:
            base_path = os.path.join(os.path.dirname(__file__), "..", "modules")

        base_path = os.path.abspath(base_path)

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    module_file = os.path.join(root, file)
                    self._register_module(module_file, base_path)

    def _register_module(self, module_file: str, base_path: str):
        """Register a single module"""
        try:
            # Calculate module path relative to base
            rel_path = os.path.relpath(module_file, base_path)
            module_path = rel_path.replace(os.sep, '/').replace('.py', '')

            # Import the module
            spec = importlib.util.spec_from_file_location(
                f"quantsploit.modules.{module_path.replace('/', '.')}",
                module_file
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Find BaseModule subclasses
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if (issubclass(obj, BaseModule) and
                    obj is not BaseModule and
                    not inspect.isabstract(obj)):

                    # Create a temporary instance to get metadata
                    instance = obj(self)
                    metadata = ModuleMetadata(
                        path=module_path,
                        name=instance.name,
                        category=instance.category,
                        description=instance.description
                    )
                    metadata.instance = obj  # Store the class, not instance

                    self.modules[module_path] = metadata
                    self.log(f"Loaded module: {module_path}", "info")

        except Exception as e:
            self.log(f"Failed to load module {module_file}: {str(e)}", "error")

    def get_module(self, module_path: str) -> Optional[Type[BaseModule]]:
        """Get a module by path"""
        if module_path in self.modules:
            return self.modules[module_path].instance
        return None

    def use_module(self, module_path: str) -> Optional[BaseModule]:
        """
        Load and activate a module for use.
        Similar to Metasploit's 'use' command.
        """
        module_class = self.get_module(module_path)
        if module_class:
            instance = module_class(self)
            self.session.load_module(instance)
            return instance
        return None

    def list_modules(self, category: Optional[str] = None) -> List[ModuleMetadata]:
        """List all available modules, optionally filtered by category"""
        modules = list(self.modules.values())
        if category:
            modules = [m for m in modules if m.category == category]
        return modules

    def search_modules(self, query: str) -> List[ModuleMetadata]:
        """Search modules by name or description"""
        query = query.lower()
        results = []
        for module in self.modules.values():
            if (query in module.name.lower() or
                query in module.description.lower() or
                query in module.path.lower()):
                results.append(module)
        return results

    def run_module(self, module: BaseModule) -> Dict:
        """Execute a module and store results"""
        valid, msg = module.validate_options()
        if not valid:
            return {"success": False, "error": msg}

        try:
            results = module.run()
            results["success"] = True

            # Store in session
            self.session.store_results(module.name, results)

            # Store in database
            symbol = module.get_option("SYMBOL")
            if symbol:
                self.database.save_analysis(
                    module.name,
                    symbol,
                    {k: v["value"] for k, v in module.options.items()},
                    results
                )

            return results

        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self.log(f"Module execution failed: {str(e)}", "error")
            return error_result

    def get_session(self) -> Session:
        """Get current session"""
        return self.session

    def get_database(self) -> Database:
        """Get database instance"""
        return self.database

    def shutdown(self):
        """Shutdown framework and cleanup"""
        if self.database:
            self.database.close()
        self.log("Framework shutdown", "info")
