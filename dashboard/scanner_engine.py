"""
Scanner Engine Helper Module

Provides startup scanning and on-demand refresh for all market scanners
via the web dashboard.
"""

import sys
from pathlib import Path

# Add parent directory to path so quantsploit module can be imported
_DASHBOARD_DIR = Path(__file__).resolve().parent
_QUANTSPLOIT_ROOT = _DASHBOARD_DIR.parent
if str(_QUANTSPLOIT_ROOT) not in sys.path:
    sys.path.insert(0, str(_QUANTSPLOIT_ROOT))

import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Scanner module mappings
SCANNER_MODULES = {
    'top_movers': ('quantsploit.modules.scanners.top_movers', 'TopMovers'),
    'price_momentum': ('quantsploit.modules.scanners.price_momentum', 'PriceMomentum'),
    'bulk_screener': ('quantsploit.modules.scanners.bulk_screener', 'BulkScreener'),
}

# Scanner metadata
SCANNER_INFO = {
    'top_movers': {
        'name': 'Top Movers',
        'description': 'Identify top gainers, losers, and momentum plays',
        'icon': 'fa-rocket',
        'default_options': {
            'SYMBOLS': 'SP500',
            'RANKING_METHOD': 'all',
            'TIMEFRAME': '1d',
            'TOP_N': 20,
            'MAX_WORKERS': 10
        }
    },
    'price_momentum': {
        'name': 'Price Momentum',
        'description': 'Find stocks with strong price momentum and trends',
        'icon': 'fa-chart-line',
        'default_options': {
            'SYMBOLS': 'SP500',
            'MIN_MOMENTUM': 5,
            'MIN_VOLUME': 1000000,
            'TOP_N': 20
        }
    },
    'bulk_screener': {
        'name': 'Bulk Screener',
        'description': 'Screen stocks based on multiple technical criteria',
        'icon': 'fa-filter',
        'default_options': {
            'SYMBOLS': 'SP500',
            'CRITERIA': 'all',
            'TOP_N': 20
        }
    }
}


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, bool):
        return obj
    elif pd.isna(obj):
        return None
    return obj


class MockFramework:
    """Mock framework for standalone scanner execution"""

    def __init__(self):
        self.database = None

    def log(self, message: str, level: str = "info"):
        """Log a message"""
        getattr(logger, level, logger.info)(message)


def get_scanner_class(scanner_id: str):
    """Dynamically import and return a scanner class"""
    if scanner_id not in SCANNER_MODULES:
        return None

    module_path, class_name = SCANNER_MODULES[scanner_id]

    try:
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name, None)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import scanner {scanner_id}: {e}")
        return None


def run_scanner(scanner_id: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute a scanner with the given options.

    Args:
        scanner_id: Scanner identifier (top_movers, price_momentum, bulk_screener)
        options: Optional dictionary of option key-value pairs

    Returns:
        Dictionary with scan results
    """
    scanner_class = get_scanner_class(scanner_id)
    if scanner_class is None:
        return {'success': False, 'error': f'Scanner {scanner_id} not found'}

    framework = MockFramework()

    try:
        instance = scanner_class(framework)

        # Get default options for this scanner
        default_opts = SCANNER_INFO.get(scanner_id, {}).get('default_options', {})

        # Merge with provided options
        merged_options = {**default_opts, **(options or {})}

        # Set options
        for key, value in merged_options.items():
            instance.set_option(key, value)

        # Execute
        logger.info(f"Running scanner: {scanner_id}")
        results = instance.run()

        # Convert to JSON-safe types
        results = convert_numpy_types(results)

        return {
            'success': True,
            'scanner_id': scanner_id,
            'scanner_name': instance.name,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.exception(f"Scanner execution failed: {e}")
        return {'success': False, 'error': str(e)}


def init_scanners(scanner_cache: Dict[str, Any]):
    """
    Initialize scanners on app startup.
    Runs all scanners in a background thread to populate the cache.

    Args:
        scanner_cache: Global cache dictionary to populate
    """
    def run_all_scanners():
        """Background thread to run all scanners"""
        logger.info("Starting background scanner initialization...")

        for scanner_id in SCANNER_MODULES.keys():
            try:
                # Mark as running
                scanner_cache[scanner_id] = {
                    'status': 'running',
                    'timestamp': None,
                    'data': None,
                    'error': None
                }

                # Run scanner with default options
                result = run_scanner(scanner_id)

                if result.get('success'):
                    scanner_cache[scanner_id] = {
                        'status': 'completed',
                        'timestamp': result.get('timestamp'),
                        'data': result.get('results'),
                        'error': None
                    }
                    logger.info(f"Scanner {scanner_id} completed successfully")
                else:
                    scanner_cache[scanner_id] = {
                        'status': 'failed',
                        'timestamp': datetime.now().isoformat(),
                        'data': None,
                        'error': result.get('error', 'Unknown error')
                    }
                    logger.warning(f"Scanner {scanner_id} failed: {result.get('error')}")

            except Exception as e:
                logger.exception(f"Error running scanner {scanner_id}: {e}")
                scanner_cache[scanner_id] = {
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat(),
                    'data': None,
                    'error': str(e)
                }

        logger.info("Background scanner initialization complete")

    # Initialize cache with pending status
    for scanner_id in SCANNER_MODULES.keys():
        scanner_cache[scanner_id] = {
            'status': 'pending',
            'timestamp': None,
            'data': None,
            'error': None
        }

    # Start background thread
    thread = threading.Thread(target=run_all_scanners, daemon=True)
    thread.start()
    logger.info("Scanner initialization thread started")


def get_scanner_info() -> Dict[str, Any]:
    """Get metadata about all available scanners"""
    return SCANNER_INFO


def refresh_scanner(scanner_id: str, scanner_cache: Dict[str, Any],
                    options: Optional[Dict[str, Any]] = None):
    """
    Refresh a specific scanner and update the cache.

    Args:
        scanner_id: Scanner to refresh
        scanner_cache: Global cache dictionary
        options: Optional custom options
    """
    def run_refresh():
        try:
            scanner_cache[scanner_id]['status'] = 'running'

            result = run_scanner(scanner_id, options)

            if result.get('success'):
                scanner_cache[scanner_id] = {
                    'status': 'completed',
                    'timestamp': result.get('timestamp'),
                    'data': result.get('results'),
                    'error': None
                }
            else:
                scanner_cache[scanner_id] = {
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat(),
                    'data': scanner_cache[scanner_id].get('data'),  # Keep old data
                    'error': result.get('error')
                }

        except Exception as e:
            scanner_cache[scanner_id] = {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'data': scanner_cache[scanner_id].get('data'),
                'error': str(e)
            }

    # Run in background thread
    thread = threading.Thread(target=run_refresh, daemon=True)
    thread.start()
