"""
Base module class for all Quantsploit modules
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime


class BaseModule(ABC):
    """
    Base class for all analysis modules in Quantsploit.
    Similar to Metasploit's module structure.
    """

    def __init__(self, framework):
        self.framework = framework
        self.options = {}
        self.results = {}
        self._init_options()

    @property
    @abstractmethod
    def name(self) -> str:
        """Module name"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Module description"""
        pass

    @property
    @abstractmethod
    def author(self) -> str:
        """Module author"""
        pass

    @property
    def category(self) -> str:
        """Module category (analysis, scanner, options, strategy)"""
        return "general"

    @property
    def required_options(self) -> List[str]:
        """List of required option keys"""
        return []

    def _init_options(self):
        """Initialize module options with defaults"""
        self.options = {
            "SYMBOL": {"value": None, "required": True, "description": "Stock/Option symbol"},
            "PERIOD": {"value": "1y", "required": False, "description": "Data period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)"},
            "INTERVAL": {"value": "1d", "required": False, "description": "Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)"},
        }

    def set_option(self, key: str, value: Any) -> bool:
        """Set an option value"""
        if key.upper() in self.options:
            self.options[key.upper()]["value"] = value
            return True
        return False

    def get_option(self, key: str) -> Any:
        """Get an option value"""
        if key.upper() in self.options:
            return self.options[key.upper()]["value"]
        return None

    def validate_options(self) -> tuple[bool, str]:
        """Validate that all required options are set"""
        for key, opt in self.options.items():
            if opt.get("required", False) and opt["value"] is None:
                return False, f"Required option '{key}' is not set"
        return True, "OK"

    def show_options(self) -> Dict[str, Any]:
        """Return formatted options for display"""
        return self.options

    def show_info(self) -> Dict[str, Any]:
        """Return module information"""
        return {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "category": self.category,
            "options": self.options
        }

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Execute the module's main functionality.
        Returns a dictionary with results.
        """
        pass

    def cleanup(self):
        """Cleanup after module execution"""
        pass

    def log(self, message: str, level: str = "info"):
        """Log a message through the framework"""
        if self.framework:
            self.framework.log(f"[{self.name}] {message}", level)


class ModuleMetadata:
    """Metadata for module registration"""

    def __init__(self, path: str, name: str, category: str, description: str):
        self.path = path
        self.name = name
        self.category = category
        self.description = description
        self.loaded = False
        self.instance = None
