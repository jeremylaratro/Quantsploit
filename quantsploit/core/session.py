"""
Session management for Quantsploit
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json


class Session:
    """
    Manages user session state, loaded modules, and workspace data
    """

    def __init__(self):
        self.current_module = None
        self.module_history = []
        self.workspace = {}
        self.variables = {}
        self.created_at = datetime.now()
        self.command_history = []

    def load_module(self, module):
        """Load a module into the current session"""
        if self.current_module:
            self.module_history.append({
                "module": self.current_module.name,
                "unloaded_at": datetime.now()
            })
        self.current_module = module

    def unload_module(self):
        """Unload the current module"""
        if self.current_module:
            self.module_history.append({
                "module": self.current_module.name,
                "unloaded_at": datetime.now()
            })
            self.current_module = None

    def set_variable(self, key: str, value: Any):
        """Set a session variable"""
        self.variables[key] = value

    def get_variable(self, key: str) -> Any:
        """Get a session variable"""
        return self.variables.get(key)

    def store_results(self, module_name: str, results: Dict[str, Any]):
        """Store module execution results"""
        if module_name not in self.workspace:
            self.workspace[module_name] = []

        self.workspace[module_name].append({
            "timestamp": datetime.now().isoformat(),
            "results": results
        })

    def get_results(self, module_name: str) -> list:
        """Get stored results for a module"""
        return self.workspace.get(module_name, [])

    def add_command(self, command: str):
        """Add command to history"""
        self.command_history.append({
            "command": command,
            "timestamp": datetime.now()
        })

    def clear_workspace(self):
        """Clear workspace data"""
        self.workspace.clear()

    def export_session(self) -> Dict[str, Any]:
        """Export session data"""
        return {
            "created_at": self.created_at.isoformat(),
            "current_module": self.current_module.name if self.current_module else None,
            "variables": self.variables,
            "workspace": self.workspace,
            "command_history": [
                {"cmd": ch["command"], "ts": ch["timestamp"].isoformat()}
                for ch in self.command_history
            ]
        }
