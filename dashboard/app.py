#!/usr/bin/env python3
"""
Comprehensive Backtesting Dashboard for Quantsploit
Real-time visualization and analysis of backtesting results
"""

from flask import Flask, render_template, jsonify, request
import json
import markdown
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from ticker_universe import (
    get_universe, get_sector, get_all_sectors, get_sector_tickers,
    get_market_cap_class, get_all_universes
)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# Disable caching for API responses
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Path to backtest results
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'backtest_results'
print(f"[STARTUP] RESULTS_DIR set to: {RESULTS_DIR}")
print(f"[STARTUP] Directory exists: {RESULTS_DIR.exists()}")


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

            # Parse timestamp (handle both YYYYmmdd_HHMMSS and YYYYmmdd_HHMMSS_counter formats)
            try:
                # Split timestamp parts (e.g., ['20231115', '101530'] or ['20231115', '101530', '1'])
                timestamp_parts = timestamp.split('_')
                if len(timestamp_parts) >= 2:
                    # Extract base timestamp (YYYYmmdd_HHMMSS)
                    base_timestamp = '_'.join(timestamp_parts[:2])  # YYYYmmdd_HHMMSS
                    dt = datetime.strptime(base_timestamp, '%Y%m%d_%H%M%S')

                    # If there's a counter suffix, append it to the display
                    if len(timestamp_parts) > 2:
                        display_time = f"{dt.strftime('%Y-%m-%d %H:%M:%S')} (#{timestamp_parts[2]})"
                    else:
                        display_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    continue

                # Check if JSON summary exists
                json_file = self.results_dir / f'summary_{timestamp}.json'
                has_json = json_file.exists()

                runs.append({
                    'timestamp': timestamp,
                    'datetime': display_time,
                    'has_json': has_json,
                    'summary_file': str(json_file) if has_json else None,
                    'csv_file': str(csv_file),
                    'report_file': str(self.results_dir / f'report_{timestamp}.md')
                })
                seen_timestamps.add(timestamp)
            except (ValueError, IndexError):
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

    def get_risk_analytics(self, timestamp: str) -> Dict:
        """Advanced risk analytics - VaR, CVaR, tail risk, etc."""
        df = self.load_detailed_results(timestamp)
        if df is None:
            return {}

        returns = df['total_return'].values

        # Value at Risk (VaR) calculations
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # Conditional VaR (CVaR/Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99

        # Drawdown statistics
        drawdowns = df['max_drawdown'].values
        avg_drawdown = np.mean(drawdowns)
        max_drawdown = np.max(drawdowns)

        # Tail risk metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Downside risk
        downside_returns = returns[returns < 0]
        downside_freq = len(downside_returns) / len(returns) * 100

        # Risk-adjusted metrics by strategy
        strategy_risk = []
        for strategy in df['strategy_name'].unique():
            strat_data = df[df['strategy_name'] == strategy]
            strategy_risk.append({
                'strategy': strategy,
                'avg_return': strat_data['total_return'].mean(),
                'volatility': strat_data['total_return'].std(),
                'sharpe': strat_data['sharpe_ratio'].mean(),
                'sortino': strat_data['sortino_ratio'].mean() if 'sortino_ratio' in df.columns else None,
                'max_dd': strat_data['max_drawdown'].max(),
                'var_95': np.percentile(strat_data['total_return'], 5),
                'downside_risk': strat_data['total_return'][strat_data['total_return'] < 0].std()
            })

        return convert_numpy_types({
            'portfolio_risk': {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'avg_drawdown': avg_drawdown,
                'max_drawdown': max_drawdown,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'downside_frequency': downside_freq
            },
            'strategy_risk': strategy_risk,
            'return_distribution': {
                'mean': np.mean(returns),
                'median': np.median(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentiles': {
                    '1': np.percentile(returns, 1),
                    '5': np.percentile(returns, 5),
                    '25': np.percentile(returns, 25),
                    '75': np.percentile(returns, 75),
                    '95': np.percentile(returns, 95),
                    '99': np.percentile(returns, 99)
                }
            }
        })

    def get_correlation_analysis(self, timestamp: str) -> Dict:
        """Correlation matrix and factor analysis"""
        df = self.load_detailed_results(timestamp)
        if df is None:
            return {}

        # Create pivot table: strategies x symbols with returns
        pivot = df.pivot_table(
            index='symbol',
            columns='strategy_name',
            values='total_return',
            aggfunc='mean'
        )

        # Correlation matrix
        corr_matrix = pivot.corr()

        # Symbol correlation (transpose)
        symbol_pivot = df.pivot_table(
            index='strategy_name',
            columns='symbol',
            values='total_return',
            aggfunc='mean'
        )
        symbol_corr = symbol_pivot.corr()

        # Find highly correlated pairs (> 0.7)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'strategy1': corr_matrix.columns[i],
                        'strategy2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })

        return convert_numpy_types({
            'strategy_correlation': {
                'matrix': corr_matrix.to_dict(),
                'strategies': corr_matrix.columns.tolist()
            },
            'symbol_correlation': {
                'matrix': symbol_corr.to_dict(),
                'symbols': symbol_corr.columns.tolist()
            },
            'high_correlations': high_corr_pairs
        })

    def get_time_period_breakdown(self, timestamp: str) -> Dict:
        """Detailed time period analysis with monthly/quarterly views"""
        df = self.load_detailed_results(timestamp)
        if df is None:
            return {}

        # Period-level analysis
        period_stats = []
        for period in df['period_name'].unique():
            period_data = df[df['period_name'] == period]

            period_stats.append({
                'period': period,
                'avg_return': period_data['total_return'].mean(),
                'median_return': period_data['total_return'].median(),
                'std_return': period_data['total_return'].std(),
                'sharpe': period_data['sharpe_ratio'].mean(),
                'win_rate': period_data['win_rate'].mean(),
                'total_trades': period_data['total_trades'].sum(),
                'best_strategy': period_data.loc[period_data['total_return'].idxmax(), 'strategy_name'],
                'worst_strategy': period_data.loc[period_data['total_return'].idxmin(), 'strategy_name'],
                'strategies_profitable': (period_data['total_return'] > 0).sum(),
                'total_strategies': len(period_data)
            })

        return convert_numpy_types({
            'period_breakdown': period_stats,
            'period_comparison': {
                'best_period': max(period_stats, key=lambda x: x['avg_return'])['period'] if period_stats else None,
                'worst_period': min(period_stats, key=lambda x: x['avg_return'])['period'] if period_stats else None,
                'most_consistent': min(period_stats, key=lambda x: x['std_return'])['period'] if period_stats else None,
                'highest_sharpe': max(period_stats, key=lambda x: x['sharpe'])['period'] if period_stats else None
            }
        })

    def get_sector_analysis(self, timestamp: str) -> Dict:
        """Sector-based performance analysis"""
        df = self.load_detailed_results(timestamp)
        if df is None:
            return {}

        # Add sector classification to each symbol
        df['sector'] = df['symbol'].apply(get_sector)

        # Aggregate by sector
        sector_stats = []
        for sector in df['sector'].unique():
            if sector == 'Unknown':
                continue

            sector_data = df[df['sector'] == sector]

            sector_stats.append({
                'sector': sector,
                'avg_return': sector_data['total_return'].mean(),
                'median_return': sector_data['total_return'].median(),
                'sharpe': sector_data['sharpe_ratio'].mean(),
                'win_rate': sector_data['win_rate'].mean(),
                'num_symbols': sector_data['symbol'].nunique(),
                'num_strategies': sector_data['strategy_name'].nunique(),
                'total_tests': len(sector_data),
                'best_symbol': sector_data.loc[sector_data['total_return'].idxmax(), 'symbol'] if len(sector_data) > 0 else None,
                'best_return': sector_data['total_return'].max()
            })

        # Sort by average return
        sector_stats.sort(key=lambda x: x['avg_return'], reverse=True)

        return convert_numpy_types({
            'sector_performance': sector_stats,
            'best_sector': sector_stats[0]['sector'] if sector_stats else None,
            'worst_sector': sector_stats[-1]['sector'] if sector_stats else None
        })

    def get_portfolio_construction(self, timestamp: str, num_strategies: int = 5) -> Dict:
        """Portfolio construction analysis - best multi-strategy portfolios"""
        df = self.load_detailed_results(timestamp)
        if df is None:
            return {}

        # Get top strategies by Sharpe ratio
        top_strategies = df.groupby('strategy_name').agg({
            'total_return': 'mean',
            'sharpe_ratio': 'mean',
            'win_rate': 'mean',
            'max_drawdown': 'mean'
        }).sort_values('sharpe_ratio', ascending=False).head(num_strategies)

        # Calculate portfolio statistics (equal weight)
        portfolio_return = top_strategies['total_return'].mean()
        portfolio_sharpe = top_strategies['sharpe_ratio'].mean()
        portfolio_max_dd = top_strategies['max_drawdown'].mean()

        # Strategy weights (can be optimized)
        strategies_list = []
        for strategy in top_strategies.index:
            strategies_list.append({
                'strategy': strategy,
                'weight': 1.0 / num_strategies,  # Equal weight
                'return': top_strategies.loc[strategy, 'total_return'],
                'sharpe': top_strategies.loc[strategy, 'sharpe_ratio'],
                'contribution': top_strategies.loc[strategy, 'total_return'] / num_strategies
            })

        # Diversification benefit
        individual_avg_return = top_strategies['total_return'].mean()
        individual_avg_sharpe = top_strategies['sharpe_ratio'].mean()

        return convert_numpy_types({
            'portfolio': {
                'expected_return': portfolio_return,
                'sharpe_ratio': portfolio_sharpe,
                'max_drawdown': portfolio_max_dd,
                'num_strategies': num_strategies
            },
            'strategies': strategies_list,
            'diversification': {
                'return_improvement': portfolio_return - individual_avg_return,
                'sharpe_improvement': portfolio_sharpe - individual_avg_sharpe
            }
        })

    def get_ticker_filtering(self, timestamp: str, filters: Dict) -> Dict:
        """Advanced ticker filtering by sector, market cap, performance"""
        df = self.load_detailed_results(timestamp)
        if df is None:
            return {}

        # Add metadata
        df['sector'] = df['symbol'].apply(get_sector)
        df['market_cap'] = df['symbol'].apply(get_market_cap_class)

        # Apply filters
        filtered_df = df.copy()

        if 'sector' in filters and filters['sector']:
            filtered_df = filtered_df[filtered_df['sector'] == filters['sector']]

        if 'market_cap' in filters and filters['market_cap']:
            filtered_df = filtered_df[filtered_df['market_cap'] == filters['market_cap']]

        if 'min_sharpe' in filters:
            filtered_df = filtered_df[filtered_df['sharpe_ratio'] >= filters['min_sharpe']]

        if 'min_return' in filters:
            filtered_df = filtered_df[filtered_df['total_return'] >= filters['min_return']]

        if 'min_win_rate' in filters:
            filtered_df = filtered_df[filtered_df['win_rate'] >= filters['min_win_rate']]

        # Return filtered results
        return convert_numpy_types({
            'total_results': len(filtered_df),
            'filters_applied': filters,
            'results': filtered_df.nlargest(100, 'total_return').to_dict('records')
        })

    def get_rolling_metrics(self, timestamp: str, window: int = 3) -> Dict:
        """Calculate rolling performance metrics across periods"""
        df = self.load_detailed_results(timestamp)
        if df is None:
            return {}

        # Sort by period and calculate rolling metrics per strategy
        rolling_data = []

        for strategy in df['strategy_name'].unique():
            strategy_data = df[df['strategy_name'] == strategy].copy()

            # Sort by period
            periods = sorted(strategy_data['period_name'].unique())

            if len(periods) >= window:
                for i in range(len(periods) - window + 1):
                    window_periods = periods[i:i+window]
                    window_data = strategy_data[strategy_data['period_name'].isin(window_periods)]

                    rolling_data.append({
                        'strategy': strategy,
                        'periods': window_periods,
                        'window': f"{window_periods[0]} to {window_periods[-1]}",
                        'avg_return': window_data['total_return'].mean(),
                        'avg_sharpe': window_data['sharpe_ratio'].mean(),
                        'consistency': window_data['total_return'].std(),
                        'win_rate': window_data['win_rate'].mean()
                    })

        return convert_numpy_types({
            'rolling_metrics': rolling_data,
            'window_size': window
        })


# Initialize data loader
data_loader = DashboardDataLoader(RESULTS_DIR)


@app.after_request
def add_no_cache_headers(response):
    """Add no-cache headers to all responses to ensure fresh data"""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    """Main dashboard page"""
    runs = data_loader.get_available_runs()
    latest_timestamp = runs[0]['timestamp'] if runs else None

    # Debug: Print to console
    print(f"DEBUG: Found {len(runs)} backtest runs")
    for i, run in enumerate(runs):
        print(f"  {i}: {run['timestamp']} - {run['datetime']}")

    return render_template('index.html', runs=runs, latest_timestamp=latest_timestamp)

@app.route('/debug/runs')
def debug_runs():
    """Debug endpoint to see what runs are being found"""
    runs = data_loader.get_available_runs()
    return f"<h1>Found {len(runs)} runs:</h1><pre>{json.dumps(runs, indent=2)}</pre>"

@app.route('/docs')
def docs():
    """Display the ticker reference documentation"""
    tickers_file = Path(__file__).parent / 'tickers.md'
    with open(tickers_file, 'r') as f:
        md_content = f.read()
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    return render_template('docs.html', content=html_content)

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


# ===== NEW INSTITUTIONAL-GRADE ANALYTICS ENDPOINTS =====

@app.route('/risk-analytics/<timestamp>')
def risk_analytics(timestamp):
    """Risk Analytics Dashboard"""
    return render_template('risk_analytics.html', timestamp=timestamp)


@app.route('/api/risk-analytics/<timestamp>')
def api_risk_analytics(timestamp):
    """API: Advanced risk analytics"""
    data = data_loader.get_risk_analytics(timestamp)
    return jsonify(data)


@app.route('/correlation/<timestamp>')
def correlation(timestamp):
    """Correlation Analysis Dashboard"""
    return render_template('correlation.html', timestamp=timestamp)


@app.route('/api/correlation/<timestamp>')
def api_correlation(timestamp):
    """API: Correlation matrix and analysis"""
    data = data_loader.get_correlation_analysis(timestamp)
    return jsonify(data)


@app.route('/time-analysis/<timestamp>')
def time_analysis(timestamp):
    """Time Period Analysis Dashboard"""
    return render_template('time_analysis.html', timestamp=timestamp)


@app.route('/api/time-analysis/<timestamp>')
def api_time_analysis(timestamp):
    """API: Detailed time period breakdown"""
    data = data_loader.get_time_period_breakdown(timestamp)
    return jsonify(data)


@app.route('/sector-analysis/<timestamp>')
def sector_analysis(timestamp):
    """Sector Analysis Dashboard"""
    return render_template('sector_analysis.html', timestamp=timestamp)


@app.route('/api/sector-analysis/<timestamp>')
def api_sector_analysis(timestamp):
    """API: Sector-based performance analysis"""
    data = data_loader.get_sector_analysis(timestamp)
    return jsonify(data)


@app.route('/portfolio/<timestamp>')
def portfolio(timestamp):
    """Portfolio Construction Dashboard"""
    return render_template('portfolio.html', timestamp=timestamp)


@app.route('/api/portfolio/<timestamp>')
def api_portfolio(timestamp):
    """API: Portfolio construction analysis"""
    num_strategies = request.args.get('num_strategies', 5, type=int)
    data = data_loader.get_portfolio_construction(timestamp, num_strategies)
    return jsonify(data)


@app.route('/ticker-explorer')
@app.route('/ticker-explorer/<timestamp>')
def ticker_explorer(timestamp=None):
    """Ticker Explorer with Advanced Filtering"""
    universes = get_all_universes()
    sectors = get_all_sectors()
    return render_template('ticker_explorer.html',
                         universes=universes,
                         sectors=sectors)


@app.route('/api/filter/<timestamp>')
def api_filter(timestamp):
    """API: Advanced ticker filtering"""
    filters = {
        'sector': request.args.get('sector'),
        'market_cap': request.args.get('market_cap'),
        'min_sharpe': request.args.get('min_sharpe', type=float),
        'min_return': request.args.get('min_return', type=float),
        'min_win_rate': request.args.get('min_win_rate', type=float)
    }
    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}

    data = data_loader.get_ticker_filtering(timestamp, filters)
    return jsonify(data)


@app.route('/rolling-metrics/<timestamp>')
def rolling_metrics(timestamp):
    """Rolling Metrics Dashboard"""
    return render_template('rolling_metrics.html', timestamp=timestamp)


@app.route('/api/rolling-metrics/<timestamp>')
def api_rolling_metrics(timestamp):
    """API: Rolling performance metrics"""
    window = request.args.get('window', 3, type=int)
    data = data_loader.get_rolling_metrics(timestamp, window)
    return jsonify(data)


@app.route('/api/universes')
def api_universes():
    """API: Get all available ticker universes"""
    return jsonify(get_all_universes())


@app.route('/api/universe/<universe_name>')
def api_universe(universe_name):
    """API: Get tickers in a specific universe"""
    tickers = get_universe(universe_name)
    return jsonify({'universe': universe_name, 'tickers': tickers, 'count': len(tickers)})


@app.route('/api/sectors')
def api_sectors():
    """API: Get all sectors"""
    return jsonify({'sectors': get_all_sectors()})


@app.route('/api/sector-tickers/<path:sector>')
def api_sector_tickers(sector):
    """API: Get tickers in a specific sector"""
    from urllib.parse import unquote
    # Explicitly URL-decode the sector name to handle special characters
    sector_decoded = unquote(sector)
    tickers = get_sector_tickers(sector_decoded)
    return jsonify({'sector': sector_decoded, 'tickers': tickers, 'count': len(tickers)})


@app.route('/api/strategies/available')
def api_strategies_available():
    """API: Get list of available trading strategies"""
    strategies = [
        {'id': 'sma_crossover', 'name': 'SMA Crossover', 'description': 'Moving average crossover (20/50)'},
        {'id': 'mean_reversion', 'name': 'Mean Reversion', 'description': '20-day mean reversion with z-score'},
        {'id': 'momentum_signals', 'name': 'Momentum', 'description': 'Multi-period momentum (10/20/50)'},
        {'id': 'multifactor_scoring', 'name': 'Multi-Factor Scoring', 'description': 'Composite factor scoring'},
        {'id': 'kalman_adaptive', 'name': 'Kalman Adaptive', 'description': 'Kalman filter-based adaptive strategy'},
        {'id': 'kalman_adaptive_sensitive', 'name': 'Kalman Adaptive (Sensitive)', 'description': 'More responsive Kalman variant'},
        {'id': 'volume_profile_swing', 'name': 'Volume Profile Swing', 'description': 'Volume profile analysis'},
        {'id': 'volume_profile_fast', 'name': 'Volume Profile (Fast)', 'description': 'Faster volume profile variant'},
        {'id': 'hmm_regime_detection', 'name': 'HMM Regime Detection', 'description': 'Hidden Markov Model regime detection'},
        {'id': 'hmm_regime_longterm', 'name': 'HMM Regime (Long-term)', 'description': 'Long-term HMM variant'},
        {'id': 'ml_swing_trading', 'name': 'ML Swing Trading', 'description': 'Machine learning swing strategy'},
        {'id': 'pairs_trading', 'name': 'Pairs Trading', 'description': 'Statistical arbitrage pairs trading'}
    ]
    return jsonify({'strategies': strategies})


@app.route('/backtest-launcher')
def backtest_launcher():
    """Backtest Launcher Page"""
    universes = get_all_universes()
    # Return sectors as a dict with display name as key and sector name as value
    sectors_list = get_all_sectors()
    sectors = {sector: sector for sector in sectors_list}
    return render_template('backtest_launcher.html',
                         universes=universes,
                         sectors=sectors)


@app.route('/api/launch-backtest', methods=['POST'])
def api_launch_backtest():
    """API: Launch a new backtest"""
    import subprocess
    import threading
    from datetime import datetime
    import uuid

    data = request.json

    # Generate job ID
    job_id = str(uuid.uuid4())[:8]

    # Build command
    cmd = ['python', '-m', 'quantsploit.main', 'use', 'analysis/comprehensive_strategy_backtest']

    # Add set commands for options
    tickers = ','.join(data['tickers'][:50])  # Limit to 50 tickers for safety

    def run_backtest():
        """Run backtest in background thread"""
        try:
            # Build comprehensive command
            from pathlib import Path
            import sys

            # Add quantsploit to path
            quantsploit_path = Path(__file__).parent.parent
            sys.path.insert(0, str(quantsploit_path))

            # Import and run backtest directly
            from quantsploit.utils.comprehensive_backtest import run_comprehensive_analysis

            # Build backtest config
            symbols = data['tickers']
            strategies = data.get('strategies', [])  # Get selected strategies
            period_config = data.get('period_config', {})
            initial_capital = data.get('initial_capital', 1000)  # Default to $1000

            # Log the received initial capital
            backtest_jobs[job_id] = {
                'status': 'running',
                'progress': 0,
                'log': f'Starting backtest with initial capital: ${initial_capital:,.2f}\n',
                'start_time': datetime.now().isoformat()
            }

            # Prepare keyword arguments (only parameters that run_comprehensive_analysis accepts)
            kwargs = {
                'symbols': symbols,
                'output_dir': str(RESULTS_DIR),
                'initial_capital': initial_capital
            }

            # Add strategies filter if specified
            if strategies:
                kwargs['strategy_keys'] = strategies
                backtest_jobs[job_id]['log'] += f'Using {len(strategies)} selected strategies\n'
            else:
                backtest_jobs[job_id]['log'] += 'Using all available strategies\n'

            # Add period configuration
            if period_config.get('mode') == 'custom':
                kwargs['tspan'] = period_config.get('tspan')
                kwargs['bspan'] = period_config.get('bspan')
                kwargs['num_periods'] = period_config.get('num_periods', 4)
            elif period_config.get('mode') == 'quarterly':
                quarters_str = ','.join(period_config.get('quarters', ['2']))
                kwargs['quarters'] = quarters_str
                # Pass years_back as num_periods for quarterly mode
                if 'years_back' in period_config:
                    kwargs['num_periods'] = period_config.get('years_back', 2)

            # Note: commission and quick_mode are not yet supported by the comprehensive backtest function

            backtest_jobs[job_id]['log'] += f'Testing {len(symbols)} symbols...\n'
            backtest_jobs[job_id]['progress'] = 10

            # Run the analysis
            run_comprehensive_analysis(**kwargs)

            backtest_jobs[job_id]['status'] = 'completed'
            backtest_jobs[job_id]['progress'] = 100
            backtest_jobs[job_id]['log'] += '\nBacktest completed successfully!'

        except Exception as e:
            backtest_jobs[job_id]['status'] = 'failed'
            backtest_jobs[job_id]['error'] = str(e)
            backtest_jobs[job_id]['log'] += f'\nError: {str(e)}'

    # Start background thread
    thread = threading.Thread(target=run_backtest, daemon=True)
    thread.start()

    return jsonify({'success': True, 'job_id': job_id})


@app.route('/api/backtest-status/<job_id>')
def api_backtest_status(job_id):
    """API: Get status of a running backtest"""
    if job_id in backtest_jobs:
        return jsonify(backtest_jobs[job_id])
    else:
        return jsonify({'status': 'not_found'}), 404


@app.route('/candlestick/<timestamp>/<strategy_name>/<symbol>')
def candlestick_view(timestamp, strategy_name, symbol):
    """Candlestick chart view for a specific strategy and symbol"""
    return render_template('candlestick.html',
                         timestamp=timestamp,
                         strategy_name=strategy_name,
                         symbol=symbol)


@app.route('/api/candlestick/<timestamp>/<strategy_name>/<symbol>')
def api_candlestick(timestamp, strategy_name, symbol):
    """API: Get candlestick data with trade signals overlaid"""
    from quantsploit.utils.data_fetcher import DataFetcher

    try:
        # Load trade details
        trades_file = RESULTS_DIR / f'trades_{timestamp}.csv'
        if not trades_file.exists():
            return jsonify({'error': 'Trade data not found. Please run a new backtest to generate trade details.'}), 404

        trades_df = pd.read_csv(trades_file)

        # Filter trades for this strategy and symbol
        strategy_trades = trades_df[
            (trades_df['strategy_name'] == strategy_name) &
            (trades_df['symbol'] == symbol)
        ]

        if len(strategy_trades) == 0:
            return jsonify({'error': 'No trades found for this strategy/symbol combination'}), 404

        # Get the date range from trades
        strategy_trades['entry_date'] = pd.to_datetime(strategy_trades['entry_date'])
        strategy_trades['exit_date'] = pd.to_datetime(strategy_trades['exit_date'])

        start_date = strategy_trades['entry_date'].min()
        end_date = strategy_trades['exit_date'].max()

        # Add buffer around dates
        from datetime import timedelta
        start_date = start_date - timedelta(days=30)
        end_date = end_date + timedelta(days=30)

        # Fetch OHLC data
        data_fetcher = DataFetcher()
        ohlc_data = data_fetcher.get_stock_data(
            symbol=symbol,
            period='3y',  # Get enough data
            interval='1d'
        )

        if ohlc_data is None or len(ohlc_data) == 0:
            return jsonify({'error': 'Failed to fetch stock data'}), 500

        # Filter to date range
        ohlc_data = ohlc_data.loc[start_date.strftime('%Y-%m-%d'):end_date.strftime('%Y-%m-%d')]

        # Prepare OHLC data for Plotly
        ohlc_data_dict = {
            'dates': ohlc_data.index.strftime('%Y-%m-%d').tolist(),
            'open': ohlc_data['Open'].tolist(),
            'high': ohlc_data['High'].tolist(),
            'low': ohlc_data['Low'].tolist(),
            'close': ohlc_data['Close'].tolist(),
            'volume': ohlc_data['Volume'].tolist() if 'Volume' in ohlc_data.columns else []
        }

        # Prepare trade data
        trades_list = []
        for _, trade in strategy_trades.iterrows():
            trades_list.append({
                'entry_date': trade['entry_date'].strftime('%Y-%m-%d'),
                'exit_date': trade['exit_date'].strftime('%Y-%m-%d'),
                'entry_price': float(trade['entry_price']),
                'exit_price': float(trade['exit_price']),
                'shares': int(trade['shares']),
                'side': trade['side'],
                'pnl': float(trade['pnl']),
                'pnl_pct': float(trade['pnl_pct']),
                'mae': float(trade['mae']),
                'mfe': float(trade['mfe'])
            })

        return jsonify(convert_numpy_types({
            'ohlc': ohlc_data_dict,
            'trades': trades_list,
            'symbol': symbol,
            'strategy': strategy_name
        }))

    except Exception as e:
        import traceback
        print(f"Error in candlestick API: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# Global dict to store backtest job status
backtest_jobs = {}


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
