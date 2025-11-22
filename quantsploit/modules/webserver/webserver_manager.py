"""
Webserver Management Module
Manage the backtesting analytics dashboard web server
"""

import subprocess
import os
import signal
import sys
from pathlib import Path
import time
from rich.console import Console
from rich.table import Table

console = Console()


class WebserverManager:
    """Standalone manager for the backtesting analytics dashboard web server"""

    def __init__(self):
        self.port = "5000"
        self.host = "127.0.0.1"
        self.action = None

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

    def start(self, port: str = None, host: str = None):
        """Start the webserver in background"""
        if port:
            self.port = port
        if host:
            self.host = host

        self._ensure_dirs()

        if self._is_running():
            console.print("\n[bold red]✗ Webserver is already running[/bold red]")
            console.print(f"  [cyan]PID:[/cyan] {self._get_pid()}")
            console.print(f"  [cyan]URL:[/cyan] http://{self.host}:{self.port}\n")
            return False

        # Get paths
        project_root = Path(__file__).parent.parent.parent.parent
        dashboard_dir = project_root / 'dashboard'
        app_file = dashboard_dir / 'app.py'

        if not app_file.exists():
            console.print(f"\n[bold red]✗ Dashboard app not found[/bold red]")
            console.print(f"  [red]Expected at: {app_file}[/red]\n")
            return False

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
            log_file.write(f"Host: {self.host}, Port: {self.port}\n")
            log_file.write(f"{'='*60}\n\n")
            log_file.flush()

            # Start the Flask app
            process = subprocess.Popen(
                [sys.executable, str(app_file), '--host', self.host, '--port', self.port, '--production'],
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
                console.print("\n[bold red]✗ Webserver failed to start[/bold red]")
                console.print(f"  [red]Check log file: {self.log_file}[/red]\n")
                return False

            # Save PID
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))

            console.print("\n[bold green]✓ Webserver Started Successfully[/bold green]\n")
            console.print(f"  [cyan]PID:[/cyan] {process.pid}")
            console.print(f"  [cyan]URL:[/cyan] http://{self.host if self.host != '0.0.0.0' else 'localhost'}:{self.port}")
            console.print(f"  [cyan]Log:[/cyan] {self.log_file}")
            console.print("\n[yellow]Tip:[/yellow] Use 'webserver status' to check status")
            console.print("[yellow]Tip:[/yellow] Use 'webserver stop' to stop the server\n")
            return True

        except Exception as e:
            console.print(f"\n[bold red]✗ Failed to start webserver[/bold red]")
            console.print(f"  [red]{str(e)}[/red]\n")
            return False

    def stop(self):
        """Stop the webserver"""
        if not self._is_running():
            console.print("\n[bold yellow]Webserver is not running[/bold yellow]\n")
            return False

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

            console.print(f"\n[bold green]✓ Webserver Stopped[/bold green]")
            console.print(f"  [cyan]PID:[/cyan] {pid}\n")
            return True

        except ProcessLookupError:
            # Process already dead
            if self.pid_file.exists():
                self.pid_file.unlink()
            console.print("\n[bold yellow]Webserver was not running[/bold yellow]\n")
            return True
        except Exception as e:
            console.print(f"\n[bold red]✗ Failed to stop webserver[/bold red]")
            console.print(f"  [red]{str(e)}[/red]\n")
            return False

    def status(self):
        """Get webserver status"""
        running = self._is_running()
        pid = self._get_pid() if running else None

        table = Table(title="Webserver Status", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green" if running else "red")

        table.add_row("Status", "Running" if running else "Stopped")
        if running:
            table.add_row("PID", str(pid))
            table.add_row("Host", self.host)
            table.add_row("Port", self.port)
            table.add_row("URL", f"http://{self.host if self.host != '0.0.0.0' else 'localhost'}:{self.port}")
        if self.log_file.exists():
            table.add_row("Log File", str(self.log_file))

        console.print()
        console.print(table)
        console.print()
        return running

    def restart(self, port: str = None, host: str = None):
        """Restart the webserver"""
        # Stop if running
        if self._is_running():
            self.stop()
            time.sleep(1)

        # Start
        return self.start(port, host)
