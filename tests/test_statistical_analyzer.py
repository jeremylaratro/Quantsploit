"""
Unit tests for Statistical Analyzer

Tests cover:
- RobustStatistics calculations
- Stratified statistics
- Outlier detection
- Group comparison
- Data quality scoring
- Ranking with confidence
- Report formatting
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.utils.statistical_analyzer import (
    StatisticalAnalyzer,
    RobustStatistics,
    StratifiedStatistics,
    StrategyRiskClass,
    STRATEGY_RISK_MAP,
    format_statistics_report,
    format_stratified_report
)


@pytest.fixture
def analyzer():
    """Create a StatisticalAnalyzer instance"""
    return StatisticalAnalyzer(min_sample_size=5, outlier_threshold=3.0)


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    return pd.Series(np.random.normal(10, 2, 100))


@pytest.fixture
def backtest_df():
    """Generate sample backtest DataFrame"""
    np.random.seed(42)
    strategies = [
        'SMA Crossover (20/50)',
        'Mean Reversion (20 day)',
        'Momentum (10/20/50)',
        'Multi-Factor Scoring',
        'Kalman Adaptive Filter',
        'HMM Regime Detection'
    ]

    data = []
    for _ in range(50):
        strategy = np.random.choice(strategies)
        data.append({
            'strategy_name': strategy,
            'total_return': np.random.normal(10, 15),
            'sharpe_ratio': np.random.normal(1.0, 0.5),
            'total_trades': np.random.randint(5, 50),
            'max_drawdown': np.random.uniform(5, 30)
        })

    return pd.DataFrame(data)


class TestRobustStatistics:
    """Tests for robust statistics calculation"""

    def test_calculate_robust_stats_basic(self, analyzer, sample_data):
        """Test basic robust statistics calculation"""
        stats = analyzer.calculate_robust_stats(sample_data)

        assert isinstance(stats, RobustStatistics)
        assert stats.count == len(sample_data)
        assert stats.mean != 0
        assert stats.std > 0

    def test_calculate_robust_stats_values(self, analyzer):
        """Test specific values are calculated correctly"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        stats = analyzer.calculate_robust_stats(data)

        assert abs(stats.mean - 5.5) < 0.01
        assert abs(stats.median - 5.5) < 0.01
        assert stats.min == 1.0
        assert stats.max == 10.0
        assert stats.count == 10

    def test_quartiles(self, analyzer):
        """Test quartile calculations"""
        data = pd.Series(range(1, 101))  # 1 to 100
        stats = analyzer.calculate_robust_stats(data)

        assert stats.q25 <= stats.median <= stats.q75
        assert stats.iqr == stats.q75 - stats.q25

    def test_empty_data(self, analyzer):
        """Test handling of empty data"""
        stats = analyzer.calculate_robust_stats(pd.Series([], dtype=float))

        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.std == 0.0

    def test_nan_handling(self, analyzer):
        """Test handling of NaN values"""
        data = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10])
        stats = analyzer.calculate_robust_stats(data)

        assert stats.count == 8  # Excludes NaN values

    def test_outlier_detection(self, analyzer):
        """Test outlier detection"""
        # Data with clear outliers
        data = pd.Series([1, 2, 3, 4, 5, 100, 200])
        stats = analyzer.calculate_robust_stats(data)

        assert stats.num_outliers > 0
        assert stats.outlier_ratio > 0

    def test_confidence_intervals(self, analyzer, sample_data):
        """Test confidence interval calculation"""
        stats = analyzer.calculate_robust_stats(sample_data)

        # CI should contain the mean
        assert stats.ci_lower <= stats.mean <= stats.ci_upper
        # CI should be reasonable range
        assert stats.ci_lower < stats.ci_upper

    def test_trimmed_mean(self, analyzer):
        """Test trimmed mean is less sensitive to outliers"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000])  # One extreme outlier
        stats = analyzer.calculate_robust_stats(data)

        # Trimmed mean should be much closer to median than mean
        assert abs(stats.trimmed_mean - stats.median) < abs(stats.mean - stats.median)

    def test_coefficient_of_variation(self, analyzer):
        """Test coefficient of variation calculation"""
        # Low variability data
        low_cv_data = pd.Series([100, 101, 99, 100, 102, 98, 100])
        low_stats = analyzer.calculate_robust_stats(low_cv_data)

        # High variability data
        high_cv_data = pd.Series([50, 150, 25, 175, 100])
        high_stats = analyzer.calculate_robust_stats(high_cv_data)

        assert low_stats.cv < high_stats.cv


class TestStratifiedStatistics:
    """Tests for stratified statistics calculation"""

    def test_calculate_stratified_stats(self, analyzer, backtest_df):
        """Test stratified statistics calculation"""
        strat_stats = analyzer.calculate_stratified_stats(
            backtest_df,
            value_col='total_return',
            strategy_col='strategy_name'
        )

        assert isinstance(strat_stats, StratifiedStatistics)
        assert strat_stats.num_samples == len(backtest_df)
        assert strat_stats.num_strategies > 0

    def test_by_risk_class(self, analyzer, backtest_df):
        """Test statistics are calculated for each risk class"""
        strat_stats = analyzer.calculate_stratified_stats(
            backtest_df,
            value_col='total_return',
            strategy_col='strategy_name'
        )

        # Should have at least some risk classes populated
        assert len(strat_stats.by_risk_class) > 0

    def test_data_quality_score(self, analyzer, backtest_df):
        """Test data quality score is calculated"""
        strat_stats = analyzer.calculate_stratified_stats(
            backtest_df,
            value_col='total_return',
            strategy_col='strategy_name'
        )

        assert 0 <= strat_stats.data_quality_score <= 100

    def test_reliability_rating(self, analyzer, backtest_df):
        """Test reliability rating is assigned"""
        strat_stats = analyzer.calculate_stratified_stats(
            backtest_df,
            value_col='total_return',
            strategy_col='strategy_name'
        )

        assert strat_stats.reliability_rating in ['High', 'Medium', 'Low']


class TestFilterValidResults:
    """Tests for result filtering"""

    def test_filter_by_min_trades(self, analyzer, backtest_df):
        """Test filtering by minimum trade count"""
        # Set min_sample_size high
        analyzer.min_sample_size = 20

        filtered = analyzer.filter_valid_results(
            backtest_df,
            min_trades_col='total_trades',
            remove_outliers=False
        )

        assert len(filtered) <= len(backtest_df)
        assert (filtered['total_trades'] >= 20).all()

    def test_filter_removes_outliers(self, analyzer):
        """Test that outlier removal works"""
        data = pd.DataFrame({
            'total_return': [10, 11, 12, 13, 14, 1000, -500],  # Two outliers
            'total_trades': [10] * 7
        })

        analyzer.min_sample_size = 5
        filtered = analyzer.filter_valid_results(
            data,
            remove_outliers=True,
            value_col='total_return'
        )

        assert len(filtered) < len(data)


class TestGroupComparison:
    """Tests for group comparison"""

    def test_compare_groups_mannwhitneyu(self, analyzer):
        """Test Mann-Whitney U comparison"""
        np.random.seed(42)
        group1 = pd.Series(np.random.normal(10, 2, 30))
        group2 = pd.Series(np.random.normal(15, 2, 30))

        stat, p_val, interp = analyzer.compare_groups(group1, group2)

        assert stat > 0
        assert 0 <= p_val <= 1
        assert isinstance(interp, str)

    def test_compare_groups_ttest(self, analyzer):
        """Test t-test comparison"""
        np.random.seed(42)
        group1 = pd.Series(np.random.normal(10, 2, 30))
        group2 = pd.Series(np.random.normal(15, 2, 30))

        stat, p_val, interp = analyzer.compare_groups(group1, group2, test='ttest')

        assert 0 <= p_val <= 1

    def test_compare_groups_insufficient_data(self, analyzer):
        """Test comparison with insufficient data"""
        group1 = pd.Series([1])
        group2 = pd.Series([2])

        stat, p_val, interp = analyzer.compare_groups(group1, group2)

        assert interp == "Insufficient data"

    def test_compare_groups_no_difference(self, analyzer):
        """Test comparison with no significant difference"""
        np.random.seed(42)
        group1 = pd.Series(np.random.normal(10, 2, 30))
        group2 = pd.Series(np.random.normal(10.1, 2, 30))  # Nearly identical

        stat, p_val, interp = analyzer.compare_groups(group1, group2)

        # p-value should be high when groups are similar
        assert p_val > 0.1 or 'No significant' in interp or 'Marginally' in interp


class TestRankWithConfidence:
    """Tests for ranking with confidence"""

    def test_rank_with_confidence_basic(self, analyzer, backtest_df):
        """Test basic ranking"""
        result = analyzer.rank_with_confidence(
            backtest_df,
            rank_by='total_return',
            group_by='strategy_name',
            top_n=5
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5

    def test_rank_includes_ci(self, analyzer, backtest_df):
        """Test ranking includes confidence intervals"""
        result = analyzer.rank_with_confidence(
            backtest_df,
            rank_by='total_return',
            group_by='strategy_name'
        )

        assert 'total_return_ci_lower' in result.columns
        assert 'total_return_ci_upper' in result.columns

    def test_rank_includes_consistency(self, analyzer, backtest_df):
        """Test ranking includes consistency score"""
        result = analyzer.rank_with_confidence(
            backtest_df,
            rank_by='total_return',
            group_by='strategy_name'
        )

        assert 'consistency' in result.columns
        assert (result['consistency'] >= 0).all()
        assert (result['consistency'] <= 1).all()

    def test_rank_sorted_descending(self, analyzer, backtest_df):
        """Test results are sorted descending by metric"""
        result = analyzer.rank_with_confidence(
            backtest_df,
            rank_by='total_return',
            group_by='strategy_name'
        )

        means = result['total_return_mean'].values
        assert all(means[i] >= means[i+1] for i in range(len(means)-1))


class TestStrategyRiskMap:
    """Tests for strategy risk class mapping"""

    def test_conservative_strategies(self):
        """Test conservative strategies are mapped correctly"""
        assert STRATEGY_RISK_MAP['SMA Crossover (20/50)'] == StrategyRiskClass.CONSERVATIVE
        assert STRATEGY_RISK_MAP['Mean Reversion (20 day)'] == StrategyRiskClass.CONSERVATIVE

    def test_moderate_strategies(self):
        """Test moderate strategies are mapped correctly"""
        assert STRATEGY_RISK_MAP['Momentum (10/20/50)'] == StrategyRiskClass.MODERATE
        assert STRATEGY_RISK_MAP['Multi-Factor Scoring'] == StrategyRiskClass.MODERATE

    def test_aggressive_strategies(self):
        """Test aggressive strategies are mapped correctly"""
        assert STRATEGY_RISK_MAP['Kalman Adaptive Filter'] == StrategyRiskClass.AGGRESSIVE
        assert STRATEGY_RISK_MAP['HMM Regime Detection'] == StrategyRiskClass.AGGRESSIVE


class TestReportFormatting:
    """Tests for report formatting functions"""

    def test_format_statistics_report(self, analyzer, sample_data):
        """Test statistics report formatting"""
        stats = analyzer.calculate_robust_stats(sample_data)
        report = format_statistics_report(stats, "Test Metric")

        assert isinstance(report, str)
        assert "Test Metric" in report
        assert "Mean" in report
        assert "Median" in report
        assert "Std Dev" in report

    def test_format_stratified_report(self, analyzer, backtest_df):
        """Test stratified report formatting"""
        strat_stats = analyzer.calculate_stratified_stats(
            backtest_df,
            value_col='total_return',
            strategy_col='strategy_name'
        )
        report = format_stratified_report(strat_stats, "Return")

        assert isinstance(report, str)
        assert "Return" in report
        assert "STRATIFIED" in report
        assert "RISK CLASS" in report


class TestEdgeCases:
    """Tests for edge cases"""

    def test_single_value(self, analyzer):
        """Test with single value"""
        stats = analyzer.calculate_robust_stats(pd.Series([5.0]))

        assert stats.count == 1
        assert stats.mean == 5.0
        assert stats.median == 5.0

    def test_all_same_values(self, analyzer):
        """Test with all identical values"""
        data = pd.Series([5.0] * 20)
        stats = analyzer.calculate_robust_stats(data)

        assert stats.mean == 5.0
        assert stats.std == 0.0
        assert stats.mad == 0.0

    def test_negative_values(self, analyzer):
        """Test with negative values"""
        data = pd.Series([-10, -5, 0, 5, 10])
        stats = analyzer.calculate_robust_stats(data)

        assert stats.mean == 0.0
        assert stats.min == -10.0
        assert stats.max == 10.0

    def test_large_dataset(self, analyzer):
        """Test with large dataset"""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 10000))
        stats = analyzer.calculate_robust_stats(data, bootstrap_ci=False)

        # Should complete without error
        assert stats.count == 10000
        # Mean should be close to 0
        assert abs(stats.mean) < 0.1

    def test_custom_outlier_threshold(self):
        """Test with custom outlier threshold"""
        # Strict threshold
        strict_analyzer = StatisticalAnalyzer(outlier_threshold=2.0)
        # Lenient threshold
        lenient_analyzer = StatisticalAnalyzer(outlier_threshold=5.0)

        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 50])  # One moderate outlier

        strict_stats = strict_analyzer.calculate_robust_stats(data)
        lenient_stats = lenient_analyzer.calculate_robust_stats(data)

        # Strict should detect more outliers
        assert strict_stats.num_outliers >= lenient_stats.num_outliers


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
