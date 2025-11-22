#!/bin/bash
# Quick start script for Quantsploit Backtesting Dashboard

echo "=================================================="
echo "  Quantsploit Backtesting Dashboard Launcher"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -d "dashboard" ]; then
    echo "Error: dashboard directory not found"
    echo "Please run this script from the Quantsploit project root"
    exit 1
fi

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
cd dashboard

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing/updating dependencies..."
pip install -q -r requirements.txt

# Check if backtest results exist
cd ..
if [ ! -d "backtest_results" ] || [ -z "$(ls -A backtest_results)" ]; then
    echo ""
    echo "WARNING: No backtest results found!"
    echo "Please run a backtest first:"
    echo "  python run_comprehensive_backtest.py --symbols AAPL,MSFT"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start the dashboard
echo ""
echo "Starting dashboard..."
echo "The dashboard will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

cd dashboard
python app.py
