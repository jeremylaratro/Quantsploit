#!/usr/bin/env python3
"""
Comprehensive Backtesting Dashboard for Quantsploit
Real-time visualization and analysis of backtesting results
"""

from flask import Flask, render_template, jsonify, request
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Path to backtest results
RESULTS_DIR = Path(__file__).parent.parent / 'backtest_results'


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle NaN and infinity
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, float):
        # Handle Python float NaN/inf
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


class DashboardDataLoader:
    """Load and process backtest data for dashboard"""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self._cache = {}

    def get_available_runs(self) -> List[Dict]:
        """Get list of all available backtest runs"""
        runs = []

        # Find all summary JSON files
        for json_file in sorted(self.results_dir.glob('summary_*.json'), reverse=True):
            timestamp = json_file.stem.replace('summary_', '')

            # Parse timestamp
            try:
                dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                runs.append({
                    'timestamp': timestamp,
                    'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'summary_file': str(json_file),
                    'csv_file': str(self.results_dir / f'detailed_results_{timestamp}.csv'),
                    'report_file': str(self.results_dir / f'report_{timestamp}.md')
                })
            except ValueError:
                continue

        return runs

    def load_summary(self, timestamp: str) -> Optional[Dict]:
        """Load summary JSON for a specific run"""
        summary_file = self.results_dir / f'summary_{timestamp}.json'

        if not summary_file.exists():
            return None

        with open(summary_file, 'r') as f:
            return json.load(f)

    def load_detailed_results(self, timestamp: str) -> Optional[pd.DataFrame]:
        """Load detailed CSV results for a specific run"""
        csv_file = self.results_dir / f'detailed_results_{timestamp}.csv'

        if not csv_file.exists():
            return None

        df = pd.read_csv(csv_file)
        return df

    def get_latest_run(self) -> Optional[str]:
        """Get timestamp of latest backtest run"""
        runs = self.get_available_runs()
        return runs[0]['timestamp'] if runs else None

    def get_quarterly_comparison(self, timestamp: str) -> Dict:
        """Generate quarterly comparison data"""
        df = self.load_detailed_results(timestamp)
        if df is None:
            return {}

        # Extract quarter info from period_name if available
        quarterly_data = []

        for period in df['period_name'].unique():
            period_data = df[df['period_name'] == period]

            quarterly_data.append({
                'period': period,
                'avg_return': period_data['total_return'].mean(),
                'avg_sharpe': period_data['sharpe_ratio'].mean(),
                'avg_win_rate': period_data['win_rate'].mean(),
                'best_strategy': period_data.loc[period_data['total_return'].idxmax(), 'strategy_name'] if len(period_data) > 0 else None,
                'best_return': period_data['total_return'].max(),
                'total_trades': period_data['total_trades'].sum()
            })

        return convert_numpy_types({
            'periods': quarterly_data,
            'period_names': df['period_name'].unique().tolist()
        })

    def get_strategy_comparison(self, timestamp: str) -> Dict:
        """Generate strategy comparison data"""
        df = self.load_detailed_results(timestamp)
        if df is None:
            return {}

        strategy_stats = []

        for strategy in df['strategy_name'].unique():
            strategy_data = df[df['strategy_name'] == strategy]

            strategy_stats.append({
                'strategy': strategy,
                'avg_return': strategy_data['total_return'].mean(),
                'avg_sharpe': strategy_data['sharpe_ratio'].mean(),
                'avg_win_rate': strategy_data['win_rate'].mean(),
                'total_trades': strategy_data['total_trades'].sum(),
                'max_return': strategy_data['total_return'].max(),
                'min_return': strategy_data['total_return'].min(),
                'consistency': strategy_data['total_return'].std()  # Lower is more consistent
            })

        # Sort by average return
        strategy_stats.sort(key=lambda x: x['avg_return'], reverse=True)

        return convert_numpy_types({
            'strategies': strategy_stats,
            'strategy_names': df['strategy_name'].unique().tolist()
        })

    def get_symbol_performance(self, timestamp: str) -> Dict:
        """Generate per-symbol performance data"""
        df = self.load_detailed_results(timestamp)
        if df is None:
            return {}

        symbol_stats = []

        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]

            symbol_stats.append({
                'symbol': symbol,
                'avg_return': symbol_data['total_return'].mean(),
                'avg_sharpe': symbol_data['sharpe_ratio'].mean(),
                'best_strategy': symbol_data.loc[symbol_data['total_return'].idxmax(), 'strategy_name'],
                'best_return': symbol_data['total_return'].max(),
                'total_trades': symbol_data['total_trades'].sum()
            })

        return convert_numpy_types({
            'symbols': symbol_stats,
            'symbol_names': df['symbol'].unique().tolist()
        })


# Initialize data loader
data_loader = DashboardDataLoader(RESULTS_DIR)


@app.route('/')
def index():
    """Main dashboard page"""
    runs = data_loader.get_available_runs()
    latest_timestamp = runs[0]['timestamp'] if runs else None

    return render_template('index.html', runs=runs, latest_timestamp=latest_timestamp)


@app.route('/api/runs')
def api_runs():
    """API: Get all available backtest runs"""
    runs = data_loader.get_available_runs()
    return jsonify(runs)


@app.route('/api/summary/<timestamp>')
def api_summary(timestamp):
    """API: Get summary data for a specific run"""
    summary = data_loader.load_summary(timestamp)
    if summary is None:
        return jsonify({'error': 'Run not found'}), 404

    return jsonify(summary)


@app.route('/api/detailed/<timestamp>')
def api_detailed(timestamp):
    """API: Get detailed results for a specific run"""
    df = data_loader.load_detailed_results(timestamp)
    if df is None:
        return jsonify({'error': 'Run not found'}), 404

    # Convert to JSON and handle numpy types
    return jsonify(convert_numpy_types(df.to_dict(orient='records')))


@app.route('/api/quarterly/<timestamp>')
def api_quarterly(timestamp):
    """API: Get quarterly comparison data"""
    data = data_loader.get_quarterly_comparison(timestamp)
    return jsonify(data)


@app.route('/api/strategies/<timestamp>')
def api_strategies(timestamp):
    """API: Get strategy comparison data"""
    data = data_loader.get_strategy_comparison(timestamp)
    return jsonify(data)


@app.route('/api/symbols/<timestamp>')
def api_symbols(timestamp):
    """API: Get symbol performance data"""
    data = data_loader.get_symbol_performance(timestamp)
    return jsonify(data)


@app.route('/strategy/<timestamp>/<strategy_name>')
def strategy_detail(timestamp, strategy_name):
    """Strategy detail page"""
    return render_template('strategy_detail.html', timestamp=timestamp, strategy_name=strategy_name)


@app.route('/period/<timestamp>/<period_name>')
def period_detail(timestamp, period_name):
    """Period detail page"""
    return render_template('period_detail.html', timestamp=timestamp, period_name=period_name)


@app.route('/comparison')
def comparison():
    """Multi-run comparison page"""
    runs = data_loader.get_available_runs()
    return render_template('comparison.html', runs=runs)


@app.route('/api/heatmap/<timestamp>')
def api_heatmap(timestamp):
    """API: Generate heatmap data for strategy x period performance"""
    df = data_loader.load_detailed_results(timestamp)
    if df is None:
        return jsonify({'error': 'Run not found'}), 404

    # Pivot table for heatmap
    pivot = df.pivot_table(
        index='strategy_name',
        columns='period_name',
        values='total_return',
        aggfunc='mean'
    )

    return jsonify(convert_numpy_types({
        'strategies': pivot.index.tolist(),
        'periods': pivot.columns.tolist(),
        'values': pivot.values.tolist()
    }))


if __name__ == '__main__':
    import argparse
    import logging

    parser = argparse.ArgumentParser(description='Quantsploit Backtesting Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--production', action='store_true', help='Run in production mode (suppresses request logging)')
    args = parser.parse_args()

    # Configure logging based on mode
    if args.production:
        # Suppress Flask's request logging in production
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        # Only log startup info
        print(f"Quantsploit Dashboard started on http://{args.host}:{args.port}")
        print(f"Results directory: {RESULTS_DIR}")
        print(f"Available runs: {len(data_loader.get_available_runs())}")

        # Run in production mode
        app.run(debug=False, host=args.host, port=args.port, threaded=True)
    else:
        # Development mode with full output
        print("\n" + "="*60)
        print("  Quantsploit Backtesting Dashboard")
        print("="*60)
        print(f"\n  📊 Dashboard URL: http://{args.host}:{args.port}")
        print(f"  📁 Results Directory: {RESULTS_DIR}")
        print(f"  🔄 Available Runs: {len(data_loader.get_available_runs())}")
        print("\n" + "="*60 + "\n")

        app.run(debug=True, host=args.host, port=args.port)
