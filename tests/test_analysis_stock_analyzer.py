"""
Unit tests for Stock Analyzer Module

Tests cover:
- StockAnalyzer initialization
- Single stock analysis
- Strategy analysis
- Period analysis
- Stock comparison
- Quality scoring
- Summary generation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.modules.analysis.stock_analyzer import (
    StockAnalyzer,
    StockAnalysis,
    StrategyPerformance,
    PeriodPerformance
)


@pytest.fixture
def sample_backtest_df():
    """Generate sample backtest results DataFrame"""
    np.random.seed(42)

    data = []
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    strategies = ['SMA Crossover (20/50)', 'Mean Reversion (20 day)', 'Momentum (10/20/50)']
    periods = ['1mo', '3mo', '6mo', '1y']

    for symbol in symbols:
        for strategy in strategies:
            for period in periods:
                data.append({
                    'symbol': symbol,
                    'strategy_name': strategy,
                    'period_name': period,
                    'total_return': np.random.normal(10, 15),
                    'sharpe_ratio': np.random.normal(1.0, 0.5),
                    'win_rate': np.random.uniform(40, 60),
                    'volatility': np.random.uniform(15, 30),
                    'max_drawdown': np.random.uniform(5, 25),
                    'total_trades': np.random.randint(10, 50)
                })

    return pd.DataFrame(data)


@pytest.fixture
def analyzer(sample_backtest_df):
    """Create a StockAnalyzer instance"""
    return StockAnalyzer(sample_backtest_df)


@pytest.fixture
def single_stock_df():
    """Generate single stock results DataFrame"""
    np.random.seed(42)

    data = []
    strategies = ['SMA Crossover (20/50)', 'Mean Reversion (20 day)', 'Momentum (10/20/50)']
    periods = ['1mo', '3mo', '6mo', '1y']

    for strategy in strategies:
        for period in periods:
            data.append({
                'symbol': 'AAPL',
                'strategy_name': strategy,
                'period_name': period,
                'total_return': np.random.normal(10, 15),
                'sharpe_ratio': np.random.normal(1.0, 0.5),
                'win_rate': np.random.uniform(40, 60),
                'volatility': np.random.uniform(15, 30),
                'max_drawdown': np.random.uniform(5, 25),
                'total_trades': np.random.randint(10, 50)
            })

    return pd.DataFrame(data)


class TestStockAnalyzerInitialization:
    """Tests for StockAnalyzer initialization"""

    def test_init_with_dataframe(self, sample_backtest_df):
        """Test initialization with DataFrame"""
        analyzer = StockAnalyzer(sample_backtest_df)
        assert analyzer.df is not None
        assert len(analyzer.df) == len(sample_backtest_df)

    def test_init_creates_stat_analyzer(self, sample_backtest_df):
        """Test that stat_analyzer is created"""
        analyzer = StockAnalyzer(sample_backtest_df)
        assert analyzer.stat_analyzer is not None


class TestAnalyzeStock:
    """Tests for analyze_stock method"""

    def test_analyze_stock_returns_analysis(self, analyzer):
        """Test that analyze_stock returns StockAnalysis"""
        analysis = analyzer.analyze_stock('AAPL')
        assert isinstance(analysis, StockAnalysis)

    def test_analyze_stock_symbol_match(self, analyzer):
        """Test that analysis has correct symbol"""
        analysis = analyzer.analyze_stock('AAPL')
        assert analysis.symbol == 'AAPL'

    def test_analyze_stock_no_data(self, analyzer):
        """Test analyze_stock returns None for unknown symbol"""
        analysis = analyzer.analyze_stock('UNKNOWN')
        assert analysis is None

    def test_analyze_stock_insufficient_trades(self, sample_backtest_df):
        """Test analyze_stock with insufficient trades filter"""
        # Modify data to have low trade counts
        sample_backtest_df['total_trades'] = 2
        analyzer = StockAnalyzer(sample_backtest_df)

        analysis = analyzer.analyze_stock('AAPL', min_trades=5)
        assert analysis is None

    def test_analyze_stock_counts(self, analyzer):
        """Test that analysis has correct counts"""
        analysis = analyzer.analyze_stock('AAPL')

        assert analysis.total_strategies == 3  # 3 strategies
        assert analysis.total_periods == 4     # 4 periods
        assert analysis.total_backtests == 12  # 3 * 4 = 12


class TestStockAnalysisFields:
    """Tests for StockAnalysis data structure"""

    def test_overall_performance_fields(self, analyzer):
        """Test overall performance fields are populated"""
        analysis = analyzer.analyze_stock('AAPL')

        assert isinstance(analysis.overall_avg_return, float)
        assert isinstance(analysis.overall_median_return, float)
        assert isinstance(analysis.overall_best_return, float)
        assert isinstance(analysis.overall_worst_return, float)

    def test_best_combination_fields(self, analyzer):
        """Test best combination fields are populated"""
        analysis = analyzer.analyze_stock('AAPL')

        assert analysis.best_strategy is not None
        assert analysis.best_period is not None
        assert isinstance(analysis.best_combination_return, float)
        assert isinstance(analysis.best_combination_sharpe, float)

    def test_most_consistent_strategy(self, analyzer):
        """Test most consistent strategy is identified"""
        analysis = analyzer.analyze_stock('AAPL')

        assert analysis.most_consistent_strategy is not None
        assert isinstance(analysis.most_consistent_cv, float)

    def test_risk_metrics(self, analyzer):
        """Test risk metrics are calculated"""
        analysis = analyzer.analyze_stock('AAPL')

        assert isinstance(analysis.avg_volatility, float)
        assert isinstance(analysis.avg_sharpe, float)
        assert isinstance(analysis.avg_max_drawdown, float)

    def test_quality_fields(self, analyzer):
        """Test data quality fields"""
        analysis = analyzer.analyze_stock('AAPL')

        assert 0 <= analysis.data_quality_score <= 100
        assert analysis.reliability_rating in ['High', 'Medium', 'Low']


class TestStrategyRankings:
    """Tests for strategy rankings"""

    def test_strategy_rankings_list(self, analyzer):
        """Test strategy rankings is a list"""
        analysis = analyzer.analyze_stock('AAPL')
        assert isinstance(analysis.strategy_rankings, list)

    def test_strategy_rankings_type(self, analyzer):
        """Test strategy rankings contains StrategyPerformance"""
        analysis = analyzer.analyze_stock('AAPL')
        for ranking in analysis.strategy_rankings:
            assert isinstance(ranking, StrategyPerformance)

    def test_strategy_rankings_sorted(self, analyzer):
        """Test strategy rankings are sorted by avg_return descending"""
        analysis = analyzer.analyze_stock('AAPL')
        rankings = analysis.strategy_rankings

        for i in range(len(rankings) - 1):
            assert rankings[i].avg_return >= rankings[i+1].avg_return

    def test_strategy_performance_fields(self, analyzer):
        """Test StrategyPerformance has required fields"""
        analysis = analyzer.analyze_stock('AAPL')
        if analysis.strategy_rankings:
            strat = analysis.strategy_rankings[0]

            assert strat.strategy_name is not None
            assert isinstance(strat.avg_return, float)
            assert isinstance(strat.median_return, float)
            assert isinstance(strat.success_rate, float)
            assert 0 <= strat.success_rate <= 100


class TestPeriodAnalysis:
    """Tests for period analysis"""

    def test_period_analysis_list(self, analyzer):
        """Test period analysis is a list"""
        analysis = analyzer.analyze_stock('AAPL')
        assert isinstance(analysis.period_analysis, list)

    def test_period_analysis_type(self, analyzer):
        """Test period analysis contains PeriodPerformance"""
        analysis = analyzer.analyze_stock('AAPL')
        for period in analysis.period_analysis:
            assert isinstance(period, PeriodPerformance)

    def test_period_analysis_sorted(self, analyzer):
        """Test period analysis is sorted by avg_return descending"""
        analysis = analyzer.analyze_stock('AAPL')
        periods = analysis.period_analysis

        for i in range(len(periods) - 1):
            assert periods[i].avg_return >= periods[i+1].avg_return

    def test_period_performance_fields(self, analyzer):
        """Test PeriodPerformance has required fields"""
        analysis = analyzer.analyze_stock('AAPL')
        if analysis.period_analysis:
            period = analysis.period_analysis[0]

            assert period.period_name is not None
            assert period.best_strategy is not None
            assert isinstance(period.avg_return, (float, np.floating))
            assert isinstance(period.profitable_strategies, (int, np.integer))


class TestCompareStocks:
    """Tests for compare_stocks method"""

    def test_compare_stocks_returns_dataframe(self, analyzer):
        """Test compare_stocks returns DataFrame"""
        result = analyzer.compare_stocks(['AAPL', 'MSFT'])
        assert isinstance(result, pd.DataFrame)

    def test_compare_stocks_columns(self, analyzer):
        """Test compare_stocks has expected columns"""
        result = analyzer.compare_stocks(['AAPL', 'MSFT'])

        expected_columns = ['symbol', 'avg_return', 'median_return', 'best_return',
                          'avg_sharpe', 'avg_volatility', 'best_strategy',
                          'most_consistent_strategy', 'total_backtests', 'reliability']

        for col in expected_columns:
            assert col in result.columns

    def test_compare_stocks_sorted(self, analyzer):
        """Test compare_stocks results are sorted by metric"""
        result = analyzer.compare_stocks(['AAPL', 'MSFT', 'GOOGL'], metric='avg_return')

        if len(result) > 1:
            values = result['avg_return'].values
            for i in range(len(values) - 1):
                assert values[i] >= values[i+1]

    def test_compare_stocks_empty_on_no_data(self, analyzer):
        """Test compare_stocks returns empty DataFrame for unknown symbols"""
        result = analyzer.compare_stocks(['UNKNOWN1', 'UNKNOWN2'])
        assert result.empty


class TestQualityScoring:
    """Tests for quality scoring"""

    def test_quality_score_range(self, analyzer):
        """Test quality score is between 0 and 100"""
        analysis = analyzer.analyze_stock('AAPL')
        assert 0 <= analysis.data_quality_score <= 100

    def test_reliability_rating_values(self, analyzer):
        """Test reliability rating is valid"""
        analysis = analyzer.analyze_stock('AAPL')
        assert analysis.reliability_rating in ['High', 'Medium', 'Low']

    def test_more_data_higher_quality(self, sample_backtest_df):
        """Test that more data results in higher quality score"""
        # Analyze with full data
        analyzer_full = StockAnalyzer(sample_backtest_df)
        analysis_full = analyzer_full.analyze_stock('AAPL')

        # Analyze with limited data
        limited_df = sample_backtest_df[sample_backtest_df['symbol'] == 'AAPL'].head(3)
        analyzer_limited = StockAnalyzer(limited_df)
        analysis_limited = analyzer_limited.analyze_stock('AAPL')

        if analysis_limited:
            assert analysis_full.data_quality_score >= analysis_limited.data_quality_score


class TestGetStockSummary:
    """Tests for get_stock_summary method"""

    def test_get_stock_summary_returns_string(self, analyzer):
        """Test get_stock_summary returns string"""
        summary = analyzer.get_stock_summary('AAPL')
        assert isinstance(summary, str)

    def test_get_stock_summary_contains_symbol(self, analyzer):
        """Test summary contains symbol"""
        summary = analyzer.get_stock_summary('AAPL')
        assert 'AAPL' in summary

    def test_get_stock_summary_contains_sections(self, analyzer):
        """Test summary contains expected sections"""
        summary = analyzer.get_stock_summary('AAPL')

        assert 'OVERVIEW' in summary
        assert 'PERFORMANCE' in summary
        assert 'RISK METRICS' in summary
        assert 'BEST COMBINATION' in summary

    def test_get_stock_summary_unknown_symbol(self, analyzer):
        """Test summary for unknown symbol"""
        summary = analyzer.get_stock_summary('UNKNOWN')
        assert 'No data available' in summary


class TestFromCSV:
    """Tests for from_csv class method"""

    def test_from_csv_creates_analyzer(self, sample_backtest_df, tmp_path):
        """Test from_csv creates StockAnalyzer"""
        csv_path = tmp_path / "test_results.csv"
        sample_backtest_df.to_csv(csv_path, index=False)

        analyzer = StockAnalyzer.from_csv(str(csv_path))
        assert isinstance(analyzer, StockAnalyzer)
        assert len(analyzer.df) == len(sample_backtest_df)


class TestEdgeCases:
    """Tests for edge cases"""

    def test_single_strategy(self):
        """Test with only one strategy"""
        data = [{
            'symbol': 'AAPL',
            'strategy_name': 'SMA Crossover (20/50)',
            'period_name': '1y',
            'total_return': 10.5,
            'sharpe_ratio': 1.2,
            'total_trades': 20
        }]
        df = pd.DataFrame(data)
        analyzer = StockAnalyzer(df)

        analysis = analyzer.analyze_stock('AAPL')
        assert analysis is not None
        assert analysis.total_strategies == 1

    def test_single_period(self):
        """Test with only one period"""
        data = [{
            'symbol': 'AAPL',
            'strategy_name': 'SMA Crossover (20/50)',
            'period_name': '1y',
            'total_return': 10.5,
            'sharpe_ratio': 1.2,
            'total_trades': 20
        }]
        df = pd.DataFrame(data)
        analyzer = StockAnalyzer(df)

        analysis = analyzer.analyze_stock('AAPL')
        assert analysis is not None
        assert analysis.total_periods == 1

    def test_all_negative_returns(self):
        """Test with all negative returns"""
        data = []
        for i in range(5):
            data.append({
                'symbol': 'AAPL',
                'strategy_name': f'Strategy {i}',
                'period_name': '1y',
                'total_return': -10 - i,
                'sharpe_ratio': -0.5,
                'total_trades': 20
            })

        df = pd.DataFrame(data)
        analyzer = StockAnalyzer(df)

        analysis = analyzer.analyze_stock('AAPL')
        assert analysis is not None
        assert analysis.overall_avg_return < 0

    def test_missing_optional_columns(self):
        """Test with missing optional columns"""
        data = [{
            'symbol': 'AAPL',
            'strategy_name': 'SMA Crossover (20/50)',
            'period_name': '1y',
            'total_return': 10.5,
            'total_trades': 20
            # Missing: sharpe_ratio, win_rate, volatility, max_drawdown
        }]
        df = pd.DataFrame(data)
        analyzer = StockAnalyzer(df)

        analysis = analyzer.analyze_stock('AAPL')
        assert analysis is not None
        assert analysis.avg_sharpe == 0.0  # Default when column missing


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
