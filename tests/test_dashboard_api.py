"""
Tests for the dashboard API helper modules (strategy_api, scanner_engine, options_helpers)
"""

import sys
from pathlib import Path
import pytest

# Add dashboard to path
DASHBOARD_DIR = Path(__file__).parent.parent / 'dashboard'
QUANTSPLOIT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(DASHBOARD_DIR))
sys.path.insert(0, str(QUANTSPLOIT_ROOT))


class TestStrategyAPI:
    """Tests for strategy_api module"""

    def test_import_strategy_api(self):
        """Test that strategy_api can be imported"""
        from strategy_api import get_all_strategies, get_strategy_options, execute_strategy
        assert callable(get_all_strategies)
        assert callable(get_strategy_options)
        assert callable(execute_strategy)

    def test_get_all_strategies_returns_list(self):
        """Test get_all_strategies returns a list of strategy dicts"""
        from strategy_api import get_all_strategies
        strategies = get_all_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_get_all_strategies_has_required_fields(self):
        """Test each strategy has required fields"""
        from strategy_api import get_all_strategies
        strategies = get_all_strategies()
        required_fields = ['id', 'name', 'description', 'category', 'options']

        for strategy in strategies:
            for field in required_fields:
                assert field in strategy, f"Missing field '{field}' in strategy {strategy.get('id', 'unknown')}"

    def test_get_all_strategies_has_core_strategies(self):
        """Test core strategies are present"""
        from strategy_api import get_all_strategies
        strategies = get_all_strategies()
        ids = [s['id'] for s in strategies]

        core_strategies = ['sma_crossover', 'mean_reversion', 'momentum_signals', 'multifactor_scoring']
        for strategy_id in core_strategies:
            assert strategy_id in ids, f"Core strategy '{strategy_id}' not found"

    def test_get_all_strategies_has_v020_strategies(self):
        """Test v0.2.0 strategies are present"""
        from strategy_api import get_all_strategies
        strategies = get_all_strategies()
        ids = [s['id'] for s in strategies]

        v020_strategies = ['risk_parity', 'volatility_breakout', 'fama_french',
                          'earnings_momentum', 'adaptive_allocation', 'options_vol_arb', 'vwap_execution']
        for strategy_id in v020_strategies:
            assert strategy_id in ids, f"v0.2.0 strategy '{strategy_id}' not found"

    def test_get_all_strategies_categories(self):
        """Test strategies have correct categories"""
        from strategy_api import get_all_strategies
        strategies = get_all_strategies()

        categories = set(s['category'] for s in strategies)
        assert 'core' in categories
        assert 'v0.2.0' in categories

    def test_get_strategy_options_core(self):
        """Test get_strategy_options for a core strategy"""
        from strategy_api import get_strategy_options
        result = get_strategy_options('sma_crossover')

        assert result is not None
        assert result['strategy_id'] == 'sma_crossover'
        assert 'name' in result
        assert 'options' in result

    def test_get_strategy_options_v020(self):
        """Test get_strategy_options for a v0.2.0 strategy"""
        from strategy_api import get_strategy_options
        result = get_strategy_options('risk_parity')

        assert result is not None
        assert result['strategy_id'] == 'risk_parity'
        assert 'name' in result
        assert 'options' in result
        assert 'SYMBOLS' in result['options']

    def test_get_strategy_options_not_found(self):
        """Test get_strategy_options returns None for unknown strategy"""
        from strategy_api import get_strategy_options
        result = get_strategy_options('nonexistent_strategy')
        assert result is None

    def test_get_strategy_categories(self):
        """Test get_strategy_categories returns expected structure"""
        from strategy_api import get_strategy_categories
        categories = get_strategy_categories()

        assert 'core' in categories
        assert 'advanced' in categories
        assert 'v0.2.0' in categories
        assert isinstance(categories['core'], list)


class TestScannerEngine:
    """Tests for scanner_engine module"""

    def test_import_scanner_engine(self):
        """Test that scanner_engine can be imported"""
        from scanner_engine import get_scanner_info, run_scanner
        assert callable(get_scanner_info)
        assert callable(run_scanner)

    def test_get_scanner_info(self):
        """Test get_scanner_info returns expected scanners"""
        from scanner_engine import get_scanner_info
        info = get_scanner_info()

        assert 'top_movers' in info
        assert 'price_momentum' in info
        assert 'bulk_screener' in info

    def test_get_scanner_info_has_required_fields(self):
        """Test each scanner has required metadata fields"""
        from scanner_engine import get_scanner_info
        info = get_scanner_info()

        for scanner_id, scanner_info in info.items():
            assert 'name' in scanner_info, f"Scanner {scanner_id} missing 'name'"
            assert 'description' in scanner_info, f"Scanner {scanner_id} missing 'description'"
            assert 'default_options' in scanner_info, f"Scanner {scanner_id} missing 'default_options'"

    def test_run_scanner_invalid_id(self):
        """Test run_scanner returns error for invalid scanner"""
        from scanner_engine import run_scanner
        result = run_scanner('nonexistent_scanner')

        assert result['success'] is False
        assert 'error' in result


class TestOptionsHelpers:
    """Tests for options_helpers module"""

    def test_import_options_helpers(self):
        """Test that options_helpers can be imported"""
        from options_helpers import (
            get_options_chain, classify_moneyness, interpret_pcr,
            suggest_strategies, calculate_single_greeks
        )
        assert callable(get_options_chain)
        assert callable(classify_moneyness)
        assert callable(interpret_pcr)
        assert callable(suggest_strategies)

    def test_classify_moneyness_call(self):
        """Test classify_moneyness for calls"""
        from options_helpers import classify_moneyness

        # Call ITM: strike < spot * 0.98
        assert classify_moneyness(90, 100, 'call') == 'ITM'
        # Call ATM: 0.98 * spot <= strike <= 1.02 * spot
        assert classify_moneyness(100, 100, 'call') == 'ATM'
        # Call OTM: strike > spot * 1.02
        assert classify_moneyness(110, 100, 'call') == 'OTM'

    def test_classify_moneyness_put(self):
        """Test classify_moneyness for puts"""
        from options_helpers import classify_moneyness

        # Put ITM: strike > spot * 1.02
        assert classify_moneyness(110, 100, 'put') == 'ITM'
        # Put ATM: 0.98 * spot <= strike <= 1.02 * spot
        assert classify_moneyness(100, 100, 'put') == 'ATM'
        # Put OTM: strike < spot * 0.98
        assert classify_moneyness(90, 100, 'put') == 'OTM'

    def test_interpret_pcr_bullish(self):
        """Test interpret_pcr for bullish reading"""
        from options_helpers import interpret_pcr
        assert 'Bullish' in interpret_pcr(0.5)

    def test_interpret_pcr_bearish(self):
        """Test interpret_pcr for bearish reading"""
        from options_helpers import interpret_pcr
        assert 'Bearish' in interpret_pcr(1.5)

    def test_interpret_pcr_neutral(self):
        """Test interpret_pcr for neutral reading"""
        from options_helpers import interpret_pcr
        assert 'Neutral' in interpret_pcr(0.85)

    def test_suggest_strategies_high_iv_bullish(self):
        """Test strategy suggestions for high IV bullish conditions"""
        from options_helpers import suggest_strategies
        suggestions = suggest_strategies('AAPL', 150.0, iv_rank=80, outlook='bullish')

        assert len(suggestions) > 0
        strategy_names = [s['name'] for s in suggestions]
        # High IV + bullish should suggest selling premium strategies
        assert any('Put' in name for name in strategy_names)

    def test_suggest_strategies_low_iv_bullish(self):
        """Test strategy suggestions for low IV bullish conditions"""
        from options_helpers import suggest_strategies
        suggestions = suggest_strategies('AAPL', 150.0, iv_rank=20, outlook='bullish')

        assert len(suggestions) > 0
        strategy_names = [s['name'] for s in suggestions]
        # Low IV + bullish should suggest buying calls
        assert any('Call' in name for name in strategy_names)

    def test_suggest_strategies_neutral(self):
        """Test strategy suggestions for neutral conditions"""
        from options_helpers import suggest_strategies
        suggestions = suggest_strategies('AAPL', 150.0, iv_rank=50, outlook='neutral')

        assert len(suggestions) > 0


class TestConvertNumpyTypes:
    """Tests for convert_numpy_types function"""

    def test_convert_numpy_int(self):
        """Test conversion of numpy integers"""
        import numpy as np
        from strategy_api import convert_numpy_types

        result = convert_numpy_types(np.int64(42))
        assert isinstance(result, int)
        assert result == 42

    def test_convert_numpy_float(self):
        """Test conversion of numpy floats"""
        import numpy as np
        from strategy_api import convert_numpy_types

        result = convert_numpy_types(np.float64(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.001

    def test_convert_numpy_nan(self):
        """Test conversion of NaN values"""
        import numpy as np
        from strategy_api import convert_numpy_types

        result = convert_numpy_types(np.float64('nan'))
        assert result is None

    def test_convert_numpy_array(self):
        """Test conversion of numpy arrays"""
        import numpy as np
        from strategy_api import convert_numpy_types

        arr = np.array([1, 2, 3])
        result = convert_numpy_types(arr)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_convert_nested_dict(self):
        """Test conversion of nested dictionaries with numpy types"""
        import numpy as np
        from strategy_api import convert_numpy_types

        data = {
            'value': np.float64(1.5),
            'nested': {'count': np.int64(10)}
        }
        result = convert_numpy_types(data)
        assert isinstance(result['value'], float)
        assert isinstance(result['nested']['count'], int)
