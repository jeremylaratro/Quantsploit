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
import markdown

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# Disable caching for API responses
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

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
        # Disable caching to always show latest results
        self._cache = {}
        self._cache_enabled = False

    def get_available_runs(self) -> List[Dict]:
        """Get list of all available backtest runs"""
        runs = []
        seen_timestamps = set()

        # Find all CSV files (primary source of truth)
        for csv_file in sorted(self.results_dir.glob('detailed_results_*.csv'), reverse=True):
            timestamp = csv_file.stem.replace('detailed_results_', '')

            if timestamp in seen_timestamps:
                continue

            # Parse timestamp
            try:
                dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')

                # Check if JSON summary exists
                json_file = self.results_dir / f'summary_{timestamp}.json'
                has_json = json_file.exists()

                runs.append({
                    'timestamp': timestamp,
                    'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'has_json': has_json,
                    'summary_file': str(json_file) if has_json else None,
                    'csv_file': str(csv_file),
                    'report_file': str(self.results_dir / f'report_{timestamp}.md')
                })
                seen_timestamps.add(timestamp)
            except ValueError:
                continue

        return runs

    def generate_summary_from_csv(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics from CSV data

        This recreates the summary dict that would normally be in the JSON file,
        allowing the dashboard to work with CSV files only.
        """
        if df.empty:
            return {"error": "No results to summarize"}

        # Helper to rank strategies by metric
        def rank_strategies(metric: str, top_n: int = 10) -> List[Dict]:
            ranked = df.nlargest(top_n, metric)
            columns = ['strategy_name', 'symbol', 'period_name', metric, 'total_return',
                       'win_rate', 'signal_accuracy', 'sharpe_ratio']
            # Remove duplicates while preserving order
            unique_columns = []
            seen = set()
            for col in columns:
                if col not in seen and col in ranked.columns:
                    unique_columns.append(col)
                    seen.add(col)
            return ranked[unique_columns].to_dict('records')

        # Helper to analyze by period
        def analyze_by_period() -> Dict:
            grouped = df.groupby('period_name').agg({
                'total_return': ['mean', 'std', 'min', 'max'],
                'sharpe_ratio': ['mean', 'std'],
                'win_rate': 'mean',
                'signal_accuracy': 'mean',
                'excess_return': 'mean'
            }).round(4)

            result = {}
            for period in grouped.index:
                result[period] = {}
                for col in grouped.columns:
                    if isinstance(col, tuple):
                        col_name = '_'.join(str(c) for c in col)
                    else:
                        col_name = str(col)
                    result[period][col_name] = float(grouped.loc[period, col])
            return result

        # Helper to analyze by symbol
        def analyze_by_symbol() -> Dict:
            grouped = df.groupby('symbol').agg({
                'total_return': ['mean', 'std'],
                'sharpe_ratio': 'mean',
                'win_rate': 'mean',
                'signal_accuracy': 'mean',
                'excess_return': 'mean'
            }).round(4)

            result = {}
            for symbol in grouped.index:
                result[symbol] = {}
                for col in grouped.columns:
                    if isinstance(col, tuple):
                        col_name = '_'.join(str(c) for c in col)
                    else:
                        col_name = str(col)
                    result[symbol][col_name] = float(grouped.loc[symbol, col])
            return result

        # Build summary
        summary = {
            'best_by_total_return': rank_strategies('total_return'),
            'best_by_sharpe_ratio': rank_strategies('sharpe_ratio'),
            'best_by_win_rate': rank_strategies('win_rate'),
            'best_by_signal_accuracy': rank_strategies('signal_accuracy'),
            'best_by_profit_factor': rank_strategies('profit_factor') if 'profit_factor' in df.columns else [],
            'best_excess_return': rank_strategies('excess_return') if 'excess_return' in df.columns else [],

            'performance_by_period': analyze_by_period(),
            'performance_by_symbol': analyze_by_symbol(),

            'overall_stats': {
                'total_backtests': len(df),
                'avg_return': float(df['total_return'].mean()),
                'avg_sharpe': float(df['sharpe_ratio'].mean()),
                'avg_win_rate': float(df['win_rate'].mean()),
                'avg_signal_accuracy': float(df['signal_accuracy'].mean()),
                'strategies_beating_buy_hold': int((df['excess_return'] > 0).sum()) if 'excess_return' in df.columns else 0,
                'percentage_beating_buy_hold': float((df['excess_return'] > 0).sum() / len(df) * 100) if 'excess_return' in df.columns else 0
            }
        }

        return convert_numpy_types(summary)

    def load_summary(self, timestamp: str) -> Optional[Dict]:
        """
        Load summary for a specific run

        First tries to load from JSON file. If JSON doesn't exist,
        generates summary from CSV data on-the-fly.
        """
        summary_file = self.results_dir / f'summary_{timestamp}.json'

        # Try to load from JSON first
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                # JSON exists but is corrupted, fall back to CSV
                print(f"Warning: Could not load JSON summary: {e}. Generating from CSV...")

        # Fall back to generating from CSV
        df = self.load_detailed_results(timestamp)
        if df is None:
            return None

        return self.generate_summary_from_csv(df)

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


@app.after_request
def add_no_cache_headers(response):
    """Add no-cache headers to all responses to ensure fresh data"""
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response


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


@app.route('/docs')
def docs():
    """Documentation page - renders tickers.md as HTML"""
    # Use absolute path to find tickers.md in project root
    tickers_file = Path(__file__).parent.parent / 'tickers.md'

    if not tickers_file.exists():
        return render_template('docs.html',
                             content='<h1>Documentation Not Found</h1><p>tickers.md file not found at project root.</p>')

    try:
        with open(tickers_file, 'r') as f:
            md_content = f.read()
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite'])
        return render_template('docs.html', content=html_content)
    except Exception as e:
        return render_template('docs.html',
                             content=f'<h1>Error Loading Documentation</h1><p>{str(e)}</p>')


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
