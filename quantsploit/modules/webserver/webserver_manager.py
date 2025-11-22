"""
Webserver Management Module
Manage the backtesting analytics dashboard web server
"""

from quantsploit.core.module import BaseModule
from typing import Dict, Any
import subprocess
import os
import signal
import sys
from pathlib import Path
import time
from rich.console import Console
from rich.table import Table

console = Console()


class WebserverManager(BaseModule):
    """Module to manage the backtesting analytics dashboard web server"""

    @property
    def name(self) -> str:
        return "Webserver Manager"

    @property
    def description(self) -> str:
        return "Start, stop, and manage the backtesting analytics dashboard"

    @property
    def author(self) -> str:
        return "Quantsploit Team"

    @property
    def category(self) -> str:
        return "webserver"

    def _init_options(self):
        super()._init_options()
        self.options.update({
            "ACTION": {
                "value": "start",
                "required": True,
                "description": "Action to perform (start/stop/status/restart)"
            },
            "PORT": {
                "value": "5000",
                "required": False,
                "description": "Port to run the webserver on"
            },
            "HOST": {
                "value": "127.0.0.1",
                "required": False,
                "description": "Host to bind to (127.0.0.1 or 0.0.0.0)"
            }
        })

    @property
    def pid_file(self) -> Path:
        """Path to PID file"""
        return Path.home() / '.quantsploit' / 'webserver.pid'

    @property
    def log_file(self) -> Path:
        """Path to log file"""
        return Path.home() / '.quantsploit' / 'webserver.log'

    def _ensure_dirs(self):
        """Ensure necessary directories exist"""
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _is_running(self) -> bool:
        """Check if webserver is running"""
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            # Check if process exists
            os.kill(pid, 0)
            return True
        except (OSError, ValueError, ProcessLookupError):
            # Process doesn't exist, clean up stale PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False

    def _get_pid(self) -> int:
        """Get PID of running webserver"""
        if not self.pid_file.exists():
            return None

        try:
            with open(self.pid_file, 'r') as f:
                return int(f.read().strip())
        except (ValueError, FileNotFoundError):
            return None

    def _start_webserver(self, port: str, host: str) -> Dict[str, Any]:
        """Start the webserver in background"""
        self._ensure_dirs()

        if self._is_running():
            return {
                "success": False,
                "message": "Webserver is already running",
                "pid": self._get_pid()
            }

        # Get paths
        project_root = Path(__file__).parent.parent.parent.parent
        dashboard_dir = project_root / 'dashboard'
        app_file = dashboard_dir / 'app.py'

        if not app_file.exists():
            return {
                "success": False,
                "message": f"Dashboard app not found at {app_file}"
            }

        # Prepare environment
        env = os.environ.copy()
        env['FLASK_ENV'] = 'production'
        env['PYTHONUNBUFFERED'] = '1'

        # Start process in background
        try:
            # Open log file for writing
            log_file = open(self.log_file, 'a')
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"Webserver started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Host: {host}, Port: {port}\n")
            log_file.write(f"{'='*60}\n\n")
            log_file.flush()

            # Start the Flask app
            process = subprocess.Popen(
                [sys.executable, str(app_file), '--host', host, '--port', port, '--production'],
                cwd=str(dashboard_dir),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True  # Detach from parent
            )

            # Give it a moment to start
            time.sleep(2)

            # Check if it's still running
            if process.poll() is not None:
                # Process died
                log_file.close()
                return {
                    "success": False,
                    "message": "Webserver failed to start. Check log file.",
                    "log_file": str(self.log_file)
                }

            # Save PID
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))

            return {
                "success": True,
                "message": "Webserver started successfully",
                "pid": process.pid,
                "url": f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}",
                "log_file": str(self.log_file)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to start webserver: {str(e)}"
            }

    def _stop_webserver(self) -> Dict[str, Any]:
        """Stop the webserver"""
        if not self._is_running():
            return {
                "success": False,
                "message": "Webserver is not running"
            }

        pid = self._get_pid()
        try:
            # Send SIGTERM
            os.kill(pid, signal.SIGTERM)

            # Wait for process to terminate
            for _ in range(10):
                time.sleep(0.5)
                if not self._is_running():
                    break
            else:
                # Force kill if still running
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            # Clean up PID file
            if self.pid_file.exists():
                self.pid_file.unlink()

            # Log shutdown
            with open(self.log_file, 'a') as f:
                f.write(f"\nWebserver stopped at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            return {
                "success": True,
                "message": "Webserver stopped successfully",
                "pid": pid
            }

        except ProcessLookupError:
            # Process already dead
            if self.pid_file.exists():
                self.pid_file.unlink()
            return {
                "success": True,
                "message": "Webserver was not running"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to stop webserver: {str(e)}"
            }

    def _get_status(self) -> Dict[str, Any]:
        """Get webserver status"""
        running = self._is_running()
        pid = self._get_pid() if running else None

        port = self.get_option("PORT")
        host = self.get_option("HOST")

        status = {
            "running": running,
            "pid": pid,
            "port": port,
            "host": host,
            "url": f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}" if running else None,
            "log_file": str(self.log_file) if self.log_file.exists() else None
        }

        return status

    def _restart_webserver(self, port: str, host: str) -> Dict[str, Any]:
        """Restart the webserver"""
        # Stop if running
        if self._is_running():
            stop_result = self._stop_webserver()
            if not stop_result["success"]:
                return stop_result
            time.sleep(1)

        # Start
        return self._start_webserver(port, host)

    def run(self) -> Dict[str, Any]:
        """Execute the webserver management action"""
        action = self.get_option("ACTION").lower()
        port = self.get_option("PORT")
        host = self.get_option("HOST")

        if action == "start":
            result = self._start_webserver(port, host)

            if result["success"]:
                console.print("\n[bold green]✓ Webserver Started Successfully[/bold green]\n")
                console.print(f"  [cyan]PID:[/cyan] {result['pid']}")
                console.print(f"  [cyan]URL:[/cyan] {result['url']}")
                console.print(f"  [cyan]Log:[/cyan] {result['log_file']}")
                console.print("\n[yellow]Tip:[/yellow] Use 'webserver status' to check status")
                console.print("[yellow]Tip:[/yellow] Use 'webserver stop' to stop the server\n")
            else:
                console.print(f"\n[bold red]✗ Failed to Start Webserver[/bold red]")
                console.print(f"  [red]{result['message']}[/red]\n")
                if "log_file" in result:
                    console.print(f"  Check log: {result['log_file']}\n")

        elif action == "stop":
            result = self._stop_webserver()

            if result["success"]:
                console.print(f"\n[bold green]✓ Webserver Stopped[/bold green]")
                console.print(f"  [cyan]PID:[/cyan] {result['pid']}\n")
            else:
                console.print(f"\n[bold red]✗ {result['message']}[/bold red]\n")

        elif action == "status":
            status = self._get_status()

            table = Table(title="Webserver Status", show_header=True)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green" if status["running"] else "red")

            table.add_row("Status", "Running" if status["running"] else "Stopped")
            if status["running"]:
                table.add_row("PID", str(status["pid"]))
                table.add_row("Host", status["host"])
                table.add_row("Port", status["port"])
                table.add_row("URL", status["url"])
            if status["log_file"]:
                table.add_row("Log File", status["log_file"])

            console.print()
            console.print(table)
            console.print()

            result = {"success": True, "status": status}

        elif action == "restart":
            result = self._restart_webserver(port, host)

            if result["success"]:
                console.print("\n[bold green]✓ Webserver Restarted Successfully[/bold green]\n")
                console.print(f"  [cyan]PID:[/cyan] {result['pid']}")
                console.print(f"  [cyan]URL:[/cyan] {result['url']}")
                console.print(f"  [cyan]Log:[/cyan] {result['log_file']}\n")
            else:
                console.print(f"\n[bold red]✗ Failed to Restart Webserver[/bold red]")
                console.print(f"  [red]{result['message']}[/red]\n")

        else:
            result = {
                "success": False,
                "message": f"Unknown action: {action}. Use start/stop/status/restart"
            }
            console.print(f"\n[bold red]✗ {result['message']}[/bold red]\n")

        return result
