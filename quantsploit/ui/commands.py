"""
Command handler for the Quantsploit console
"""

from typing import Dict, List, Optional, Callable
import shlex
from .display import Display


class CommandHandler:
    """
    Handles command parsing and execution.
    Similar to Metasploit's command structure.
    """

    def __init__(self, framework):
        self.framework = framework
        self.display = Display()
        self.commands = self._register_commands()

    def _register_commands(self) -> Dict[str, Callable]:
        """Register all available commands"""
        return {
            "help": self.cmd_help,
            "?": self.cmd_help,
            "show": self.cmd_show,
            "use": self.cmd_use,
            "back": self.cmd_back,
            "info": self.cmd_info,
            "options": self.cmd_options,
            "set": self.cmd_set,
            "unset": self.cmd_unset,
            "run": self.cmd_run,
            "exploit": self.cmd_run,  # Alias for run
            "search": self.cmd_search,
            "exit": self.cmd_exit,
            "quit": self.cmd_exit,
            "clear": self.cmd_clear,
            "history": self.cmd_history,
            "watchlist": self.cmd_watchlist,
            "quote": self.cmd_quote,
            "sessions": self.cmd_sessions,
            "webserver": self.cmd_webserver,
        }

    def get_command_descriptions(self) -> Dict[str, str]:
        """Get descriptions for all commands"""
        return {
            "help/?": "Display help information",
            "show": "Show modules, options, or other data (show modules/options/watchlist)",
            "use": "Load a module for use (use <module_path>)",
            "back": "Unload the current module",
            "info": "Display current module information",
            "options": "Show current module options",
            "set": "Set a module option (set <OPTION> <value>)",
            "unset": "Unset a module option (unset <OPTION>)",
            "run/exploit": "Execute the current module",
            "search": "Search for modules (search <query>)",
            "quote": "Get real-time quote (quote <SYMBOL>)",
            "watchlist": "Manage watchlist (watchlist add/remove/show <SYMBOL>)",
            "history": "Show command history",
            "sessions": "Show session information",
            "webserver": "Manage analytics dashboard webserver (webserver start/stop/status/restart)",
            "clear": "Clear the screen",
            "exit/quit": "Exit Quantsploit",
        }

    def execute(self, command_line: str) -> bool:
        """
        Execute a command line.
        Returns False if should exit, True otherwise.
        """
        if not command_line.strip():
            return True

        # Store in session history
        self.framework.session.add_command(command_line)

        # Parse command
        try:
            parts = shlex.split(command_line)
        except ValueError as e:
            self.display.print_error(f"Invalid command syntax: {str(e)}")
            return True

        if not parts:
            return True

        cmd = parts[0].lower()
        args = parts[1:]

        # Execute command
        if cmd in self.commands:
            return self.commands[cmd](args)
        else:
            self.display.print_error(f"Unknown command: {cmd}")
            self.display.print_info("Type 'help' for available commands")
            return True

    def cmd_help(self, args: List[str]) -> bool:
        """Display help information"""
        self.display.print_help(self.get_command_descriptions())
        return True

    def cmd_show(self, args: List[str]) -> bool:
        """Show various information"""
        if not args:
            self.display.print_error("Usage: show <modules|options|watchlist>")
            return True

        what = args[0].lower()

        if what == "modules":
            category = args[1] if len(args) > 1 else None
            modules = self.framework.list_modules(category)
            self.display.print_modules(modules, category or "All")

        elif what == "options":
            if not self.framework.session.current_module:
                self.display.print_error("No module loaded. Use 'use <module>' first")
            else:
                options = self.framework.session.current_module.show_options()
                self.display.print_options(options)

        elif what == "watchlist":
            watchlist = self.framework.database.get_watchlist()
            if watchlist:
                self.display._print_list(
                    [f"{w['symbol']} - {w['notes']}" for w in watchlist],
                    "Watchlist"
                )
            else:
                self.display.print_info("Watchlist is empty")

        else:
            self.display.print_error(f"Unknown show option: {what}")

        return True

    def cmd_use(self, args: List[str]) -> bool:
        """Load a module"""
        if not args:
            self.display.print_error("Usage: use <module_path>")
            return True

        module_path = args[0]
        module = self.framework.use_module(module_path)

        if module:
            self.display.print_success(f"Loaded module: {module.name}")
            self.display.print_info(module.description)
        else:
            self.display.print_error(f"Module not found: {module_path}")
            self.display.print_info("Use 'search' to find modules")

        return True

    def cmd_back(self, args: List[str]) -> bool:
        """Unload current module"""
        if self.framework.session.current_module:
            name = self.framework.session.current_module.name
            self.framework.session.unload_module()
            self.display.print_success(f"Unloaded module: {name}")
        else:
            self.display.print_warning("No module loaded")
        return True

    def cmd_info(self, args: List[str]) -> bool:
        """Display module information"""
        if not self.framework.session.current_module:
            self.display.print_error("No module loaded. Use 'use <module>' first")
        else:
            info = self.framework.session.current_module.show_info()
            self.display.print_module_info(info)
            self.display.print("\n")
            self.display.print_options(info['options'])
        return True

    def cmd_options(self, args: List[str]) -> bool:
        """Show module options"""
        return self.cmd_show(["options"])

    def cmd_set(self, args: List[str]) -> bool:
        """Set a module option"""
        if not self.framework.session.current_module:
            self.display.print_error("No module loaded. Use 'use <module>' first")
            return True

        if len(args) < 2:
            self.display.print_error("Usage: set <OPTION> <value>")
            return True

        option = args[0].upper()
        value = " ".join(args[1:])

        if self.framework.session.current_module.set_option(option, value):
            self.display.print_success(f"Set {option} => {value}")
        else:
            self.display.print_error(f"Unknown option: {option}")

        return True

    def cmd_unset(self, args: List[str]) -> bool:
        """Unset a module option"""
        if not self.framework.session.current_module:
            self.display.print_error("No module loaded. Use 'use <module>' first")
            return True

        if not args:
            self.display.print_error("Usage: unset <OPTION>")
            return True

        option = args[0].upper()
        if self.framework.session.current_module.set_option(option, None):
            self.display.print_success(f"Unset {option}")
        else:
            self.display.print_error(f"Unknown option: {option}")

        return True

    def cmd_run(self, args: List[str]) -> bool:
        """Execute current module"""
        if not self.framework.session.current_module:
            self.display.print_error("No module loaded. Use 'use <module>' first")
            return True

        self.display.print_info(f"Running module: {self.framework.session.current_module.name}...")

        results = self.framework.run_module(self.framework.session.current_module)
        self.display.print("\n")
        self.display.print_results(results)

        return True

    def cmd_search(self, args: List[str]) -> bool:
        """Search for modules"""
        if not args:
            self.display.print_error("Usage: search <query>")
            return True

        query = " ".join(args)
        results = self.framework.search_modules(query)

        if results:
            self.display.print_modules(results, f"Search: '{query}'")
        else:
            self.display.print_warning(f"No modules found matching '{query}'")

        return True

    def cmd_quote(self, args: List[str]) -> bool:
        """Get real-time quote"""
        if not args:
            self.display.print_error("Usage: quote <SYMBOL>")
            return True

        from quantsploit.utils.data_fetcher import DataFetcher
        fetcher = DataFetcher(self.framework.database)

        symbol = args[0].upper()
        quote = fetcher.get_realtime_quote(symbol)

        if quote:
            self.display._print_dict(quote, f"Quote: {symbol}")
        else:
            self.display.print_error(f"Failed to fetch quote for {symbol}")

        return True

    def cmd_watchlist(self, args: List[str]) -> bool:
        """Manage watchlist"""
        if not args:
            return self.cmd_show(["watchlist"])

        action = args[0].lower()

        if action == "add" and len(args) >= 2:
            symbol = args[1].upper()
            notes = " ".join(args[2:]) if len(args) > 2 else ""
            if self.framework.database.add_to_watchlist(symbol, notes):
                self.display.print_success(f"Added {symbol} to watchlist")
            else:
                self.display.print_warning(f"{symbol} already in watchlist")

        elif action == "remove" and len(args) >= 2:
            symbol = args[1].upper()
            self.framework.database.remove_from_watchlist(symbol)
            self.display.print_success(f"Removed {symbol} from watchlist")

        elif action == "show":
            return self.cmd_show(["watchlist"])

        else:
            self.display.print_error("Usage: watchlist <add|remove|show> <SYMBOL> [notes]")

        return True

    def cmd_history(self, args: List[str]) -> bool:
        """Show command history"""
        history = self.framework.session.command_history
        if history:
            self.display._print_list(
                [f"{h['command']} ({h['timestamp'].strftime('%H:%M:%S')})" for h in history],
                "Command History"
            )
        else:
            self.display.print_info("No command history")
        return True

    def cmd_sessions(self, args: List[str]) -> bool:
        """Show session information"""
        session_data = self.framework.session.export_session()
        self.display._print_dict(session_data, "Session Information")
        return True

    def cmd_clear(self, args: List[str]) -> bool:
        """Clear the screen"""
        import os
        os.system('clear' if os.name != 'nt' else 'cls')
        return True

    def cmd_webserver(self, args: List[str]) -> bool:
        """Manage analytics dashboard webserver"""
        if not args:
            self.display.print_error("Usage: webserver <start|stop|status|restart> [--port PORT] [--host HOST]")
            return True

        action = args[0].lower()

        # Parse additional arguments
        port = "5000"
        host = "127.0.0.1"

        i = 1
        while i < len(args):
            if args[i] == "--port" and i + 1 < len(args):
                port = args[i + 1]
                i += 2
            elif args[i] == "--host" and i + 1 < len(args):
                host = args[i + 1]
                i += 2
            else:
                i += 1

        # Import and run the webserver manager
        try:
            from quantsploit.modules.webserver.webserver_manager import WebserverManager

            manager = WebserverManager()
            manager.set_option("ACTION", action)
            manager.set_option("PORT", port)
            manager.set_option("HOST", host)
            manager.run()

        except ImportError as e:
            self.display.print_error("Webserver module not found")
            self.display.print_info("Make sure the dashboard is properly installed")
        except Exception as e:
            self.display.print_error(f"Webserver error: {str(e)}")

        return True

    def cmd_exit(self, args: List[str]) -> bool:
        """Exit the console"""
        self.display.print_info("Shutting down Quantsploit...")
        return False
