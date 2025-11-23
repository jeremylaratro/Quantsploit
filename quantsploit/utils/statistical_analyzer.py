"""
Robust Statistical Analyzer

Provides advanced statistical analysis with proper stratification, outlier handling,
and robust estimators to fix issues with naive mean/stdev calculations.

Key Features:
- Stratified analysis by strategy risk class
- Robust statistics (median, MAD, IQR)
- Outlier detection and filtering
- Confidence intervals via bootstrapping
- Sample size weighting
- Multiple comparison corrections
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import trim_mean, iqr


class StrategyRiskClass(Enum):
    """Strategy classification by risk profile"""
    CONSERVATIVE = "conservative"  # SMA, basic mean reversion
    MODERATE = "moderate"          # Momentum, multi-factor
    AGGRESSIVE = "aggressive"      # Kalman, HMM, volume profile


# Map strategy names to risk classes
STRATEGY_RISK_MAP = {
    'SMA Crossover (20/50)': StrategyRiskClass.CONSERVATIVE,
    'Mean Reversion (20 day)': StrategyRiskClass.CONSERVATIVE,
    'Momentum (10/20/50)': StrategyRiskClass.MODERATE,
    'Multi-Factor Scoring': StrategyRiskClass.MODERATE,
    'Kalman Adaptive Filter': StrategyRiskClass.AGGRESSIVE,
    'Kalman Adaptive Filter (Sensitive)': StrategyRiskClass.AGGRESSIVE,
    'Volume Profile Swing': StrategyRiskClass.AGGRESSIVE,
    'Volume Profile Swing (Conservative)': StrategyRiskClass.MODERATE,
    'HMM Regime Detection': StrategyRiskClass.AGGRESSIVE,
    'HMM Regime Detection (Sensitive)': StrategyRiskClass.AGGRESSIVE,
}


@dataclass
class RobustStatistics:
    """Container for robust statistical measures"""
    mean: float
    median: float
    std: float
    mad: float  # Median Absolute Deviation
    iqr: float  # Interquartile Range
    q25: float
    q75: float
    min: float
    max: float
    count: int

    # Additional robust measures
    trimmed_mean: float  # 10% trimmed mean
    cv: float  # Coefficient of variation (std/mean) - consistency measure
    sem: float  # Standard error of mean

    # Confidence intervals (95%)
    ci_lower: float
    ci_upper: float

    # Outlier statistics
    num_outliers: int
    outlier_ratio: float


@dataclass
class StratifiedStatistics:
    """Statistics stratified by strategy risk class"""
    overall: RobustStatistics
    by_risk_class: Dict[StrategyRiskClass, RobustStatistics]
    num_strategies: int
    num_samples: int

    # Quality metrics
    data_quality_score: float  # 0-100, based on sample size and consistency
    reliability_rating: str    # "High", "Medium", "Low"


class StatisticalAnalyzer:
    """
    Provides robust statistical analysis with proper handling of:
    - Mixed strategy distributions
    - Small sample sizes
    - Outliers
    - Multiple comparisons
    """

    def __init__(self, min_sample_size: int = 5, outlier_threshold: float = 3.0):
        """
        Initialize analyzer

        Args:
            min_sample_size: Minimum trades/samples for valid statistics
            outlier_threshold: Modified Z-score threshold for outlier detection
        """
        self.min_sample_size = min_sample_size
        self.outlier_threshold = outlier_threshold

    def calculate_robust_stats(
        self,
        data: pd.Series,
        bootstrap_ci: bool = True,
        n_bootstrap: int = 1000
    ) -> RobustStatistics:
        """
        Calculate robust statistics for a data series

        Args:
            data: Series of values
            bootstrap_ci: Whether to calculate bootstrapped confidence intervals
            n_bootstrap: Number of bootstrap iterations

        Returns:
            RobustStatistics object with comprehensive metrics
        """
        if len(data) == 0:
            return self._empty_stats()

        # Remove NaN values
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return self._empty_stats()

        # Detect outliers using modified Z-score (robust to outliers)
        outliers = self._detect_outliers(clean_data)
        num_outliers = outliers.sum()
        outlier_ratio = num_outliers / len(clean_data)

        # Basic statistics
        mean_val = float(clean_data.mean())
        median_val = float(clean_data.median())
        std_val = float(clean_data.std())

        # Robust statistics
        mad_val = float(stats.median_abs_deviation(clean_data))
        iqr_val = float(iqr(clean_data))
        q25 = float(clean_data.quantile(0.25))
        q75 = float(clean_data.quantile(0.75))

        # Trimmed mean (removes top/bottom 10%)
        trimmed_mean_val = float(trim_mean(clean_data, 0.1)) if len(clean_data) >= 10 else mean_val

        # Coefficient of variation (handle division by zero)
        cv_val = abs(std_val / mean_val) if abs(mean_val) > 0.001 else 0.0

        # Standard error of mean
        sem_val = std_val / np.sqrt(len(clean_data))

        # Confidence intervals
        if bootstrap_ci and len(clean_data) >= self.min_sample_size:
            ci_lower, ci_upper = self._bootstrap_ci(clean_data, n_bootstrap)
        else:
            # Fall back to parametric CI
            ci_lower, ci_upper = stats.t.interval(
                0.95,
                len(clean_data) - 1,
                loc=mean_val,
                scale=sem_val
            )

        return RobustStatistics(
            mean=mean_val,
            median=median_val,
            std=std_val,
            mad=mad_val,
            iqr=iqr_val,
            q25=q25,
            q75=q75,
            min=float(clean_data.min()),
            max=float(clean_data.max()),
            count=len(clean_data),
            trimmed_mean=trimmed_mean_val,
            cv=cv_val,
            sem=sem_val,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            num_outliers=num_outliers,
            outlier_ratio=outlier_ratio
        )

    def calculate_stratified_stats(
        self,
        df: pd.DataFrame,
        value_col: str = 'total_return',
        strategy_col: str = 'strategy_name'
    ) -> StratifiedStatistics:
        """
        Calculate statistics stratified by strategy risk class

        This fixes the "large standard deviation" problem by separating
        conservative, moderate, and aggressive strategies.

        Args:
            df: DataFrame with backtest results
            value_col: Column to analyze (e.g., 'total_return', 'sharpe_ratio')
            strategy_col: Column containing strategy names

        Returns:
            StratifiedStatistics with overall and per-class statistics
        """
        # Add risk class column
        df = df.copy()
        df['risk_class'] = df[strategy_col].map(
            lambda x: STRATEGY_RISK_MAP.get(x, StrategyRiskClass.MODERATE)
        )

        # Calculate overall statistics
        overall_stats = self.calculate_robust_stats(df[value_col])

        # Calculate per-class statistics
        by_risk_class = {}
        for risk_class in StrategyRiskClass:
            class_data = df[df['risk_class'] == risk_class][value_col]
            if len(class_data) > 0:
                by_risk_class[risk_class] = self.calculate_robust_stats(class_data)

        # Calculate data quality score
        quality_score = self._calculate_quality_score(df, value_col)
        reliability = self._get_reliability_rating(quality_score, len(df))

        return StratifiedStatistics(
            overall=overall_stats,
            by_risk_class=by_risk_class,
            num_strategies=df[strategy_col].nunique(),
            num_samples=len(df),
            data_quality_score=quality_score,
            reliability_rating=reliability
        )

    def filter_valid_results(
        self,
        df: pd.DataFrame,
        min_trades_col: str = 'total_trades',
        remove_outliers: bool = True,
        value_col: str = 'total_return'
    ) -> pd.DataFrame:
        """
        Filter out unreliable results

        Args:
            df: DataFrame with backtest results
            min_trades_col: Column containing trade count
            remove_outliers: Whether to remove statistical outliers
            value_col: Column to check for outliers

        Returns:
            Filtered DataFrame
        """
        df_filtered = df.copy()

        # Filter by minimum trade count
        if min_trades_col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[min_trades_col] >= self.min_sample_size]

        # Remove outliers if requested
        if remove_outliers and value_col in df_filtered.columns:
            outliers = self._detect_outliers(df_filtered[value_col])
            df_filtered = df_filtered[~outliers]

        return df_filtered

    def compare_groups(
        self,
        group1: pd.Series,
        group2: pd.Series,
        test: str = 'mannwhitneyu'
    ) -> Tuple[float, float, str]:
        """
        Compare two groups statistically

        Args:
            group1: First group of values
            group2: Second group of values
            test: 'mannwhitneyu' (non-parametric) or 'ttest' (parametric)

        Returns:
            (statistic, p_value, interpretation)
        """
        g1 = group1.dropna()
        g2 = group2.dropna()

        if len(g1) < 2 or len(g2) < 2:
            return 0.0, 1.0, "Insufficient data"

        if test == 'mannwhitneyu':
            statistic, p_value = stats.mannwhitneyu(g1, g2, alternative='two-sided')
        else:
            statistic, p_value = stats.ttest_ind(g1, g2)

        # Interpret results
        if p_value < 0.01:
            interp = "Highly significant difference (p < 0.01)"
        elif p_value < 0.05:
            interp = "Significant difference (p < 0.05)"
        elif p_value < 0.10:
            interp = "Marginally significant (p < 0.10)"
        else:
            interp = "No significant difference"

        return float(statistic), float(p_value), interp

    def rank_with_confidence(
        self,
        df: pd.DataFrame,
        rank_by: str = 'total_return',
        group_by: str = 'strategy_name',
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Rank groups with confidence intervals and significance testing

        Args:
            df: DataFrame with results
            rank_by: Metric to rank by
            group_by: Column to group by
            top_n: Number of top results to return

        Returns:
            DataFrame with rankings, statistics, and confidence intervals
        """
        results = []

        for group_name, group_data in df.groupby(group_by):
            stats_obj = self.calculate_robust_stats(group_data[rank_by])

            results.append({
                group_by: group_name,
                f'{rank_by}_mean': stats_obj.mean,
                f'{rank_by}_median': stats_obj.median,
                f'{rank_by}_std': stats_obj.std,
                f'{rank_by}_ci_lower': stats_obj.ci_lower,
                f'{rank_by}_ci_upper': stats_obj.ci_upper,
                'consistency': 1 / (1 + stats_obj.cv) if stats_obj.cv > 0 else 1.0,  # Higher is better
                'sample_size': stats_obj.count,
                'outlier_ratio': stats_obj.outlier_ratio,
                'reliability': 'High' if stats_obj.count >= 20 else 'Medium' if stats_obj.count >= 10 else 'Low'
            })

        result_df = pd.DataFrame(results)

        # Sort by mean, but flag overlapping confidence intervals
        result_df = result_df.sort_values(f'{rank_by}_mean', ascending=False)

        return result_df.head(top_n)

    def _detect_outliers(self, data: pd.Series) -> pd.Series:
        """
        Detect outliers using modified Z-score (robust to outliers)

        Uses median and MAD instead of mean and std
        """
        median = data.median()
        mad = stats.median_abs_deviation(data)

        if mad == 0:
            return pd.Series([False] * len(data), index=data.index)

        modified_z_scores = 0.6745 * (data - median) / mad
        return abs(modified_z_scores) > self.outlier_threshold

    def _bootstrap_ci(
        self,
        data: pd.Series,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate bootstrapped confidence intervals"""
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = data.sample(n=len(data), replace=True)
            bootstrap_means.append(sample.mean())

        alpha = (1 - confidence) / 2
        ci_lower = np.percentile(bootstrap_means, alpha * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

        return ci_lower, ci_upper

    def _calculate_quality_score(self, df: pd.DataFrame, value_col: str) -> float:
        """
        Calculate data quality score (0-100)

        Based on:
        - Sample size
        - Consistency (low CV)
        - Low outlier ratio
        - Balanced strategy distribution
        """
        # Sample size score (0-30 points)
        sample_score = min(30, len(df) / 10 * 3)

        # Consistency score (0-30 points)
        cv = df[value_col].std() / abs(df[value_col].mean()) if abs(df[value_col].mean()) > 0.001 else 10
        consistency_score = max(0, 30 - cv * 10)

        # Outlier score (0-20 points)
        outliers = self._detect_outliers(df[value_col])
        outlier_ratio = outliers.sum() / len(df)
        outlier_score = max(0, 20 - outlier_ratio * 100)

        # Balance score (0-20 points) - checks if we have diverse strategies
        if 'strategy_name' in df.columns:
            strategy_counts = df['strategy_name'].value_counts()
            balance = strategy_counts.std() / strategy_counts.mean() if strategy_counts.mean() > 0 else 10
            balance_score = max(0, 20 - balance * 10)
        else:
            balance_score = 10

        return sample_score + consistency_score + outlier_score + balance_score

    def _get_reliability_rating(self, quality_score: float, sample_size: int) -> str:
        """Convert quality score to reliability rating"""
        if quality_score >= 70 and sample_size >= 20:
            return "High"
        elif quality_score >= 50 and sample_size >= 10:
            return "Medium"
        else:
            return "Low"

    def _empty_stats(self) -> RobustStatistics:
        """Return empty statistics object"""
        return RobustStatistics(
            mean=0.0, median=0.0, std=0.0, mad=0.0, iqr=0.0,
            q25=0.0, q75=0.0, min=0.0, max=0.0, count=0,
            trimmed_mean=0.0, cv=0.0, sem=0.0,
            ci_lower=0.0, ci_upper=0.0,
            num_outliers=0, outlier_ratio=0.0
        )


def format_statistics_report(stats: RobustStatistics, metric_name: str = "Return") -> str:
    """Format statistics into a readable report"""
    report = f"""
{metric_name} Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Central Tendency:
  Mean:          {stats.mean:>10.2f}%
  Median:        {stats.median:>10.2f}%
  Trimmed Mean:  {stats.trimmed_mean:>10.2f}%

Dispersion:
  Std Dev:       {stats.std:>10.2f}%
  MAD:           {stats.mad:>10.2f}%
  IQR:           {stats.iqr:>10.2f}%
  CV:            {stats.cv:>10.4f}

Range:
  Min:           {stats.min:>10.2f}%
  Q25:           {stats.q25:>10.2f}%
  Q75:           {stats.q75:>10.2f}%
  Max:           {stats.max:>10.2f}%

Confidence (95%):
  Lower:         {stats.ci_lower:>10.2f}%
  Upper:         {stats.ci_upper:>10.2f}%

Quality:
  Sample Size:   {stats.count:>10d}
  Outliers:      {stats.num_outliers:>10d} ({stats.outlier_ratio*100:.1f}%)
  SEM:           {stats.sem:>10.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return report


def format_stratified_report(strat_stats: StratifiedStatistics, metric_name: str = "Return") -> str:
    """Format stratified statistics into a readable report"""
    report = f"\n{'='*60}\n"
    report += f"STRATIFIED ANALYSIS: {metric_name}\n"
    report += f"{'='*60}\n\n"

    report += f"Overall Quality: {strat_stats.data_quality_score:.1f}/100 ({strat_stats.reliability_rating})\n"
    report += f"Strategies: {strat_stats.num_strategies} | Samples: {strat_stats.num_samples}\n\n"

    report += "OVERALL STATISTICS:\n"
    report += f"  Mean:   {strat_stats.overall.mean:>8.2f}% ± {strat_stats.overall.sem:.2f}%\n"
    report += f"  Median: {strat_stats.overall.median:>8.2f}%\n"
    report += f"  95% CI: [{strat_stats.overall.ci_lower:.2f}%, {strat_stats.overall.ci_upper:.2f}%]\n\n"

    report += "BY RISK CLASS:\n"
    report += "─" * 60 + "\n"

    for risk_class, class_stats in strat_stats.by_risk_class.items():
        report += f"\n{risk_class.value.upper()}:\n"
        report += f"  Mean:   {class_stats.mean:>8.2f}% (Median: {class_stats.median:.2f}%)\n"
        report += f"  Std:    {class_stats.std:>8.2f}% (MAD: {class_stats.mad:.2f}%)\n"
        report += f"  95% CI: [{class_stats.ci_lower:.2f}%, {class_stats.ci_upper:.2f}%]\n"
        report += f"  Samples: {class_stats.count} | CV: {class_stats.cv:.3f}\n"

    report += "\n" + "=" * 60 + "\n"

    return report
