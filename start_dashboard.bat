@echo off
REM Quick start script for Quantsploit Backtesting Dashboard (Windows)

echo ==================================================
echo   Quantsploit Backtesting Dashboard Launcher
echo ==================================================
echo.

REM Check if we're in the right directory
if not exist "dashboard" (
    echo Error: dashboard directory not found
    echo Please run this script from the Quantsploit project root
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install dependencies
echo Checking dependencies...
cd dashboard

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing/updating dependencies...
pip install -q -r requirements.txt

REM Check if backtest results exist
cd ..
if not exist "backtest_results" (
    echo.
    echo WARNING: No backtest results found!
    echo Please run a backtest first:
    echo   python run_comprehensive_backtest.py --symbols AAPL,MSFT
    echo.
    pause
)

REM Start the dashboard
echo.
echo Starting dashboard...
echo The dashboard will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

cd dashboard
python app.py

pause
