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
            "analyze": self.cmd_analyze,
            "compare": self.cmd_compare,
            "filter": self.cmd_filter,
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
            "analyze": "Analyze stocks, sectors, or periods (analyze stock/sector/period <NAME>)",
            "compare": "Compare strategies head-to-head (compare <STRATEGY1> <STRATEGY2> [--stock SYMBOL])",
            "filter": "Filter backtest results (filter --sector AI/Tech --min-sharpe 1.0)",
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
        """Load a module with smart search"""
        if not args:
            self.display.print_error("Usage: use <module_path_or_keyword>")
            return True

        query = args[0]

        # First try exact path
        module = self.framework.use_module(query)

        if module:
            self.display.print_success(f"Loaded module: {module.name}")
            self.display.print_info(module.description)
            return True

        # If not found, try smart search by keyword
        self.display.print_info(f"Module '{query}' not found, searching for matches...")

        # Search for modules matching the keyword
        matches = self.framework.search_modules(query)

        if not matches:
            self.display.print_error(f"No modules found matching '{query}'")
            self.display.print_info("Use 'show modules' to see all available modules")
            return True

        if len(matches) == 1:
            # Only one match, load it automatically
            module_path = matches[0]['path']
            module = self.framework.use_module(module_path)
            if module:
                self.display.print_success(f"Auto-loaded: {module.name}")
                self.display.print_info(module.description)
        else:
            # Multiple matches, show options
            self.display.print_info(f"Found {len(matches)} matching modules:")
            self.display.print_modules(matches, f"Search: '{query}'")
            self.display.print_info("\nPlease specify the full module path, e.g.:")
            self.display.print_info(f"  use {matches[0]['path']}")

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
        port = None
        host = None

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

            if action == "start":
                manager.start(port, host)
            elif action == "stop":
                manager.stop()
            elif action == "status":
                manager.status()
            elif action == "restart":
                manager.restart(port, host)
            else:
                self.display.print_error(f"Unknown action: {action}")
                self.display.print_info("Use: start, stop, status, or restart")

        except ImportError as e:
            self.display.print_error("Webserver module not found")
            self.display.print_info("Make sure the dashboard is properly installed")
        except Exception as e:
            self.display.print_error(f"Webserver error: {str(e)}")
            import traceback
            traceback.print_exc()

        return True

    def cmd_analyze(self, args: List[str]) -> bool:
        """Analyze stocks, sectors, or periods from latest backtest results"""
        if not args:
            self.display.print_error("Usage: analyze <stock|sector|period> <NAME> [--timestamp TS]")
            self.display.print_info("Examples:")
            self.display.print_info("  analyze stock AAPL")
            self.display.print_info("  analyze sector AI/Tech")
            self.display.print_info("  analyze period 1yr")
            self.display.print_info("  analyze stock NVDA --timestamp 20251122_203908")
            return True

        analysis_type = args[0].lower()

        if analysis_type not in ['stock', 'sector', 'period']:
            self.display.print_error(f"Unknown analysis type: {analysis_type}")
            self.display.print_info("Use: stock, sector, or period")
            return True

        if len(args) < 2:
            self.display.print_error(f"Usage: analyze {analysis_type} <NAME>")
            return True

        name = args[1]

        # Parse optional timestamp
        timestamp = None
        if len(args) > 2 and args[2] == '--timestamp' and len(args) > 3:
            timestamp = args[3]

        try:
            from pathlib import Path

            # Find latest results if no timestamp specified
            if not timestamp:
                results_dir = Path('backtest_results')
                csv_files = list(results_dir.glob('detailed_results_*.csv'))
                if not csv_files:
                    self.display.print_error("No backtest results found!")
                    self.display.print_info("Run a backtest first using: use backtesting/comprehensive")
                    return True
                latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
                csv_path = str(latest_csv)
                self.display.print_info(f"Using latest results: {latest_csv.name}")
            else:
                csv_path = f'backtest_results/detailed_results_{timestamp}.csv'

            # Perform analysis based on type
            if analysis_type == 'stock':
                from modules.analysis.stock_analyzer import StockAnalyzer
                analyzer = StockAnalyzer.from_csv(csv_path)
                summary = analyzer.get_stock_summary(name.upper())
                self.display.print(summary)

            elif analysis_type == 'sector':
                from modules.analysis.sector_deep_dive import SectorAnalyzer
                analyzer = SectorAnalyzer.from_csv(csv_path)
                perf = analyzer.analyze_sector(name)
                if perf:
                    report = analyzer.format_sector_report(perf)
                    self.display.print(report)
                else:
                    self.display.print_error(f"No data available for sector: {name}")

            elif analysis_type == 'period':
                self.display.print_warning("Period analysis not yet implemented via CLI")
                self.display.print_info("Use the Python scripts directly for now")

        except ImportError as e:
            self.display.print_error(f"Analysis module not found: {str(e)}")
        except FileNotFoundError:
            self.display.print_error(f"Results file not found: {csv_path}")
        except Exception as e:
            self.display.print_error(f"Analysis error: {str(e)}")
            import traceback
            traceback.print_exc()

        return True

    def cmd_compare(self, args: List[str]) -> bool:
        """Compare strategies head-to-head"""
        if len(args) < 2:
            self.display.print_error("Usage: compare <STRATEGY1> <STRATEGY2> [--stock SYMBOL] [--sector SECTOR]")
            self.display.print_info("Examples:")
            self.display.print_info("  compare 'SMA Crossover (20/50)' 'Kalman Adaptive Filter'")
            self.display.print_info("  compare 'SMA Crossover (20/50)' 'Momentum (10/20/50)' --stock AAPL")
            self.display.print_info("  compare 'Multi-Factor Scoring' 'HMM Regime Detection' --sector AI/Tech")
            return True

        # Parse strategies and optional filters
        strategy1 = args[0]
        strategy2 = args[1]
        stock = None
        sector = None

        i = 2
        while i < len(args):
            if args[i] == '--stock' and i + 1 < len(args):
                stock = args[i + 1].upper()
                i += 2
            elif args[i] == '--sector' and i + 1 < len(args):
                sector = args[i + 1]
                i += 2
            else:
                i += 1

        try:
            from pathlib import Path
            from modules.analysis.strategy_comparator import StrategyComparator

            # Find latest results
            results_dir = Path('backtest_results')
            csv_files = list(results_dir.glob('detailed_results_*.csv'))
            if not csv_files:
                self.display.print_error("No backtest results found!")
                return True
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
            self.display.print_info(f"Using latest results: {latest_csv.name}\n")

            # Load comparator
            comparator = StrategyComparator.from_csv(str(latest_csv))

            # Compare
            result = comparator.compare_two_strategies(
                strategy1,
                strategy2,
                stock=stock,
                sector=sector
            )

            if result:
                report = comparator.format_comparison(result)
                self.display.print(report)
            else:
                self.display.print_error("Insufficient data for comparison")
                self.display.print_info("Make sure both strategies have been tested with enough samples")

        except ImportError as e:
            self.display.print_error(f"Comparison module not found: {str(e)}")
        except Exception as e:
            self.display.print_error(f"Comparison error: {str(e)}")
            import traceback
            traceback.print_exc()

        return True

    def cmd_filter(self, args: List[str]) -> bool:
        """Filter backtest results with multiple criteria"""
        if not args:
            self.display.print_error("Usage: filter [options]")
            self.display.print_info("\nFilter Options:")
            self.display.print_info("  --sector SECTOR        Filter by sector")
            self.display.print_info("  --symbol SYMBOL        Filter by stock symbol")
            self.display.print_info("  --strategy STRATEGY    Filter by strategy")
            self.display.print_info("  --period PERIOD        Filter by time period")
            self.display.print_info("  --min-return PCT       Minimum return threshold")
            self.display.print_info("  --min-sharpe RATIO     Minimum Sharpe ratio")
            self.display.print_info("  --min-win-rate PCT     Minimum win rate")
            self.display.print_info("  --max-volatility PCT   Maximum volatility")
            self.display.print_info("  --top-n N              Show only top N results")
            self.display.print_info("\nExamples:")
            self.display.print_info("  filter --sector AI/Tech --min-sharpe 1.0")
            self.display.print_info("  filter --symbol AAPL --min-return 10")
            self.display.print_info("  filter --strategy 'Kalman Adaptive Filter' --min-win-rate 60")
            self.display.print_info("  filter --min-sharpe 1.5 --top-n 10")
            return True

        # Parse filter arguments
        sector = None
        symbol = None
        strategy = None
        period = None
        min_return = None
        min_sharpe = None
        min_win_rate = None
        max_volatility = None
        top_n = None

        i = 0
        while i < len(args):
            if args[i] == '--sector' and i + 1 < len(args):
                sector = args[i + 1]
                i += 2
            elif args[i] == '--symbol' and i + 1 < len(args):
                symbol = args[i + 1].upper()
                i += 2
            elif args[i] == '--strategy' and i + 1 < len(args):
                strategy = args[i + 1]
                i += 2
            elif args[i] == '--period' and i + 1 < len(args):
                period = args[i + 1]
                i += 2
            elif args[i] == '--min-return' and i + 1 < len(args):
                min_return = float(args[i + 1])
                i += 2
            elif args[i] == '--min-sharpe' and i + 1 < len(args):
                min_sharpe = float(args[i + 1])
                i += 2
            elif args[i] == '--min-win-rate' and i + 1 < len(args):
                min_win_rate = float(args[i + 1])
                i += 2
            elif args[i] == '--max-volatility' and i + 1 < len(args):
                max_volatility = float(args[i + 1])
                i += 2
            elif args[i] == '--top-n' and i + 1 < len(args):
                top_n = int(args[i + 1])
                i += 2
            else:
                i += 1

        try:
            from pathlib import Path
            from modules.analysis.advanced_filter import AdvancedFilter

            # Find latest results
            results_dir = Path('backtest_results')
            csv_files = list(results_dir.glob('detailed_results_*.csv'))
            if not csv_files:
                self.display.print_error("No backtest results found!")
                return True
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
            self.display.print_info(f"Using latest results: {latest_csv.name}\n")

            # Load filter
            filter_sys = AdvancedFilter.from_csv(str(latest_csv))

            # Apply filters
            filter_kwargs = {}
            if sector:
                filter_kwargs['sector'] = sector
            if symbol:
                filter_kwargs['symbol'] = symbol
            if strategy:
                filter_kwargs['strategy'] = strategy
            if period:
                filter_kwargs['period'] = period
            if min_return is not None:
                filter_kwargs['min_return'] = min_return
            if min_sharpe is not None:
                filter_kwargs['min_sharpe'] = min_sharpe
            if min_win_rate is not None:
                filter_kwargs['min_win_rate'] = min_win_rate
            if max_volatility is not None:
                filter_kwargs['max_volatility'] = max_volatility

            if top_n:
                results_df = filter_sys.top_n(n=top_n, **filter_kwargs)
                self.display.print_info(f"Top {top_n} results:\n")
            else:
                results_df = filter_sys.quick_filter(**filter_kwargs)
                self.display.print_info(f"Found {len(results_df)} results:\n")

            if len(results_df) > 0:
                # Display results
                display_cols = ['symbol', 'strategy_name', 'period_name', 'total_return', 'sharpe_ratio', 'win_rate']
                available_cols = [c for c in display_cols if c in results_df.columns]

                self.display.print(results_df[available_cols].to_string(index=False))
            else:
                self.display.print_warning("No results match the filter criteria")

        except ImportError as e:
            self.display.print_error(f"Filter module not found: {str(e)}")
        except Exception as e:
            self.display.print_error(f"Filter error: {str(e)}")
            import traceback
            traceback.print_exc()

        return True

    def cmd_exit(self, args: List[str]) -> bool:
        """Exit the console"""
        self.display.print_info("Shutting down Quantsploit...")
        return False
