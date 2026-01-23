"""
Data Validation Pipeline for Quantsploit

This module provides comprehensive data validation, cleaning, and quality assessment
for OHLCV financial data. It includes:

- DataValidator: Validates OHLCV data integrity and detects anomalies
- DataCleaner: Cleans and normalizes data issues
- MissingDataHandler: Configurable strategies for handling missing data

All classes provide detailed logging and clear error messages for debugging.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings

# Configure module-level logger
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MissingDataStrategy(Enum):
    """Strategies for handling missing data"""
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    INTERPOLATE = "interpolate"
    DROP = "drop"
    MEAN = "mean"
    MEDIAN = "median"
    ZERO = "zero"


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in the data"""
    issue_type: str
    severity: ValidationSeverity
    message: str
    affected_rows: List[int] = field(default_factory=list)
    affected_columns: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    details: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        row_info = f" (rows: {len(self.affected_rows)})" if self.affected_rows else ""
        return f"[{self.severity.value.upper()}] {self.issue_type}: {self.message}{row_info}"


@dataclass
class QualityReport:
    """Comprehensive data quality report"""
    symbol: str = ""
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    total_rows: int = 0
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    trading_days_expected: int = 0
    trading_days_actual: int = 0
    completeness_score: float = 0.0
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"DATA QUALITY REPORT: {self.symbol}",
            f"{'='*60}",
            f"Analysis Timestamp: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Date Range: {self.date_range_start} to {self.date_range_end}",
            f"Total Rows: {self.total_rows}",
            f"Trading Days Expected: {self.trading_days_expected}",
            f"Trading Days Actual: {self.trading_days_actual}",
            f"Completeness Score: {self.completeness_score:.2%}",
            f"\n{'='*60}",
            f"ISSUES FOUND: {len(self.issues)}",
            f"{'='*60}",
        ]

        # Group issues by severity
        by_severity = {}
        for issue in self.issues:
            sev = issue.severity.value
            if sev not in by_severity:
                by_severity[sev] = []
            by_severity[sev].append(issue)

        for severity in ['critical', 'error', 'warning', 'info']:
            if severity in by_severity:
                lines.append(f"\n{severity.upper()} ({len(by_severity[severity])}):")
                for issue in by_severity[severity]:
                    lines.append(f"  - {issue}")

        if self.summary:
            lines.append(f"\n{'='*60}")
            lines.append("SUMMARY STATISTICS:")
            lines.append(f"{'='*60}")
            for key, value in self.summary.items():
                lines.append(f"  {key}: {value}")

        lines.append(f"\n{'='*60}\n")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert report to dictionary for JSON serialization"""
        return {
            "symbol": self.symbol,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "total_rows": self.total_rows,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "trading_days_expected": self.trading_days_expected,
            "trading_days_actual": self.trading_days_actual,
            "completeness_score": self.completeness_score,
            "issues_count": len(self.issues),
            "issues_by_severity": {
                "critical": sum(1 for i in self.issues if i.severity == ValidationSeverity.CRITICAL),
                "error": sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR),
                "warning": sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING),
                "info": sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO),
            },
            "issues": [
                {
                    "type": i.issue_type,
                    "severity": i.severity.value,
                    "message": i.message,
                    "affected_rows_count": len(i.affected_rows),
                    "details": i.details
                }
                for i in self.issues
            ],
            "summary": self.summary
        }


class DataValidator:
    """
    Validates OHLCV financial data for integrity, consistency, and quality.

    Performs comprehensive checks including:
    - Missing data detection (NaN values)
    - Invalid OHLC relationships (High < Low, etc.)
    - Outlier detection using statistical methods
    - Gap detection for missing trading days
    - Bad tick detection (zero prices, negative values)
    - Stale data detection
    - Duplicate timestamp detection

    Example:
        validator = DataValidator()
        report = validator.generate_quality_report(df, symbol="AAPL")
        print(report)
    """

    # Standard trading days per year (approximate)
    TRADING_DAYS_PER_YEAR = 252

    # Required OHLCV columns
    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

    def __init__(
        self,
        outlier_std_threshold: float = 3.0,
        max_daily_return: float = 0.20,
        min_volume_threshold: int = 0,
        stale_data_days: int = 5
    ):
        """
        Initialize DataValidator with configurable thresholds.

        Args:
            outlier_std_threshold: Number of standard deviations for outlier detection
            max_daily_return: Maximum expected single-day return (default 20%)
            min_volume_threshold: Minimum expected volume (0 to disable)
            stale_data_days: Number of days without change to flag as stale
        """
        self.outlier_std_threshold = outlier_std_threshold
        self.max_daily_return = max_daily_return
        self.min_volume_threshold = min_volume_threshold
        self.stale_data_days = stale_data_days

    def validate_ohlcv(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """
        Check OHLCV data integrity and structural validity.

        Validates:
        - Required columns exist
        - Data types are numeric
        - Index is datetime
        - OHLC relationships are valid (High >= Low, etc.)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of ValidationIssue objects describing any problems found
        """
        issues = []

        if df is None or len(df) == 0:
            issues.append(ValidationIssue(
                issue_type="empty_data",
                severity=ValidationSeverity.CRITICAL,
                message="DataFrame is empty or None"
            ))
            return issues

        # Check required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            issues.append(ValidationIssue(
                issue_type="missing_columns",
                severity=ValidationSeverity.CRITICAL,
                message=f"Missing required columns: {missing_cols}",
                affected_columns=missing_cols
            ))
            return issues  # Cannot continue without required columns

        # Check index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append(ValidationIssue(
                issue_type="invalid_index",
                severity=ValidationSeverity.ERROR,
                message="Index is not DatetimeIndex. Consider converting with pd.to_datetime()"
            ))

        # Check data types
        for col in self.REQUIRED_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(ValidationIssue(
                    issue_type="invalid_dtype",
                    severity=ValidationSeverity.ERROR,
                    message=f"Column '{col}' is not numeric (dtype: {df[col].dtype})",
                    affected_columns=[col]
                ))

        # Check OHLC relationships
        # High should be >= Open, Close, Low
        high_low_violation = df[df['High'] < df['Low']]
        if len(high_low_violation) > 0:
            issues.append(ValidationIssue(
                issue_type="ohlc_high_low_violation",
                severity=ValidationSeverity.ERROR,
                message=f"High < Low in {len(high_low_violation)} rows",
                affected_rows=high_low_violation.index.tolist(),
                details={"sample_dates": high_low_violation.index[:5].tolist()}
            ))

        high_open_violation = df[df['High'] < df['Open']]
        if len(high_open_violation) > 0:
            issues.append(ValidationIssue(
                issue_type="ohlc_high_open_violation",
                severity=ValidationSeverity.ERROR,
                message=f"High < Open in {len(high_open_violation)} rows",
                affected_rows=high_open_violation.index.tolist()
            ))

        high_close_violation = df[df['High'] < df['Close']]
        if len(high_close_violation) > 0:
            issues.append(ValidationIssue(
                issue_type="ohlc_high_close_violation",
                severity=ValidationSeverity.ERROR,
                message=f"High < Close in {len(high_close_violation)} rows",
                affected_rows=high_close_violation.index.tolist()
            ))

        low_open_violation = df[df['Low'] > df['Open']]
        if len(low_open_violation) > 0:
            issues.append(ValidationIssue(
                issue_type="ohlc_low_open_violation",
                severity=ValidationSeverity.ERROR,
                message=f"Low > Open in {len(low_open_violation)} rows",
                affected_rows=low_open_violation.index.tolist()
            ))

        low_close_violation = df[df['Low'] > df['Close']]
        if len(low_close_violation) > 0:
            issues.append(ValidationIssue(
                issue_type="ohlc_low_close_violation",
                severity=ValidationSeverity.ERROR,
                message=f"Low > Close in {len(low_close_violation)} rows",
                affected_rows=low_close_violation.index.tolist()
            ))

        logger.debug(f"OHLCV validation found {len(issues)} issues")
        return issues

    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str] = None
    ) -> List[ValidationIssue]:
        """
        Identify price/volume outliers using statistical methods.

        Uses z-score method: values more than N standard deviations from mean
        are flagged as outliers.

        Args:
            df: DataFrame with OHLCV data
            columns: Columns to check (default: Close and Volume)

        Returns:
            List of ValidationIssue objects for detected outliers
        """
        issues = []

        if columns is None:
            columns = ['Close', 'Volume']

        for col in columns:
            if col not in df.columns:
                continue

            # Calculate z-scores
            mean = df[col].mean()
            std = df[col].std()

            if std == 0 or pd.isna(std):
                issues.append(ValidationIssue(
                    issue_type="zero_variance",
                    severity=ValidationSeverity.WARNING,
                    message=f"Column '{col}' has zero variance",
                    affected_columns=[col]
                ))
                continue

            z_scores = (df[col] - mean) / std
            outliers = df[abs(z_scores) > self.outlier_std_threshold]

            if len(outliers) > 0:
                issues.append(ValidationIssue(
                    issue_type="statistical_outliers",
                    severity=ValidationSeverity.WARNING,
                    message=f"Found {len(outliers)} outliers in '{col}' (>{self.outlier_std_threshold} sigma)",
                    affected_rows=outliers.index.tolist(),
                    affected_columns=[col],
                    details={
                        "mean": mean,
                        "std": std,
                        "threshold": self.outlier_std_threshold,
                        "outlier_values": outliers[col].tolist()[:10]  # First 10
                    }
                ))

        # Check for excessive daily returns
        if 'Close' in df.columns:
            returns = df['Close'].pct_change(fill_method=None)
            excessive_returns = df[abs(returns) > self.max_daily_return]

            if len(excessive_returns) > 0:
                issues.append(ValidationIssue(
                    issue_type="excessive_returns",
                    severity=ValidationSeverity.WARNING,
                    message=f"Found {len(excessive_returns)} days with returns > {self.max_daily_return:.0%}",
                    affected_rows=excessive_returns.index.tolist(),
                    details={
                        "max_return_threshold": self.max_daily_return,
                        "actual_returns": returns.loc[excessive_returns.index].tolist()
                    }
                ))

        logger.debug(f"Outlier detection found {len(issues)} issues")
        return issues

    def detect_gaps(
        self,
        df: pd.DataFrame,
        max_gap_days: int = 5
    ) -> List[ValidationIssue]:
        """
        Find missing trading days in the data.

        Detects gaps larger than expected (accounting for weekends/holidays).

        Args:
            df: DataFrame with datetime index
            max_gap_days: Maximum allowed gap in calendar days

        Returns:
            List of ValidationIssue objects for detected gaps
        """
        issues = []

        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append(ValidationIssue(
                issue_type="invalid_index_for_gap_detection",
                severity=ValidationSeverity.ERROR,
                message="Cannot detect gaps: index is not DatetimeIndex"
            ))
            return issues

        if len(df) < 2:
            return issues

        # Calculate gaps between consecutive dates
        dates = df.index.sort_values()
        gaps = pd.Series(dates[1:]) - pd.Series(dates[:-1].values)

        # Find gaps larger than threshold
        large_gaps = []
        for i, gap in enumerate(gaps):
            gap_days = gap.days
            if gap_days > max_gap_days:
                large_gaps.append({
                    'start_date': dates[i],
                    'end_date': dates[i + 1],
                    'gap_days': gap_days
                })

        if large_gaps:
            issues.append(ValidationIssue(
                issue_type="data_gaps",
                severity=ValidationSeverity.WARNING,
                message=f"Found {len(large_gaps)} gaps > {max_gap_days} days",
                details={
                    "gaps": [
                        {
                            "from": g['start_date'].isoformat(),
                            "to": g['end_date'].isoformat(),
                            "days": g['gap_days']
                        }
                        for g in large_gaps[:10]  # First 10 gaps
                    ]
                }
            ))

        logger.debug(f"Gap detection found {len(issues)} issues")
        return issues

    def detect_bad_ticks(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """
        Detect bad tick data: zero prices, negative values, NaN values.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of ValidationIssue objects for bad ticks
        """
        issues = []

        price_cols = ['Open', 'High', 'Low', 'Close']

        # Check for NaN values
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                continue

            nan_rows = df[df[col].isna()]
            if len(nan_rows) > 0:
                issues.append(ValidationIssue(
                    issue_type="nan_values",
                    severity=ValidationSeverity.ERROR,
                    message=f"Found {len(nan_rows)} NaN values in '{col}'",
                    affected_rows=nan_rows.index.tolist(),
                    affected_columns=[col]
                ))

        # Check for zero prices
        for col in price_cols:
            if col not in df.columns:
                continue

            zero_rows = df[df[col] == 0]
            if len(zero_rows) > 0:
                issues.append(ValidationIssue(
                    issue_type="zero_prices",
                    severity=ValidationSeverity.ERROR,
                    message=f"Found {len(zero_rows)} zero values in '{col}'",
                    affected_rows=zero_rows.index.tolist(),
                    affected_columns=[col]
                ))

        # Check for negative prices
        for col in price_cols:
            if col not in df.columns:
                continue

            negative_rows = df[df[col] < 0]
            if len(negative_rows) > 0:
                issues.append(ValidationIssue(
                    issue_type="negative_prices",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Found {len(negative_rows)} negative values in '{col}'",
                    affected_rows=negative_rows.index.tolist(),
                    affected_columns=[col]
                ))

        # Check for zero volume on trading days
        if 'Volume' in df.columns:
            zero_volume = df[df['Volume'] == 0]
            if len(zero_volume) > 0:
                issues.append(ValidationIssue(
                    issue_type="zero_volume",
                    severity=ValidationSeverity.WARNING,
                    message=f"Found {len(zero_volume)} days with zero volume",
                    affected_rows=zero_volume.index.tolist(),
                    affected_columns=['Volume']
                ))

            # Check for negative volume
            negative_volume = df[df['Volume'] < 0]
            if len(negative_volume) > 0:
                issues.append(ValidationIssue(
                    issue_type="negative_volume",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Found {len(negative_volume)} negative volume values",
                    affected_rows=negative_volume.index.tolist(),
                    affected_columns=['Volume']
                ))

        logger.debug(f"Bad tick detection found {len(issues)} issues")
        return issues

    def detect_stale_data(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """
        Detect stale data (no price changes over extended periods).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of ValidationIssue objects for stale data periods
        """
        issues = []

        if 'Close' not in df.columns or len(df) < self.stale_data_days:
            return issues

        # Find periods where close price doesn't change
        price_changes = df['Close'].diff().fillna(1)
        unchanged_mask = price_changes == 0

        # Find consecutive unchanged periods
        stale_periods = []
        consecutive_count = 0
        start_idx = None

        for i, is_unchanged in enumerate(unchanged_mask):
            if is_unchanged:
                if start_idx is None:
                    start_idx = i
                consecutive_count += 1
            else:
                if consecutive_count >= self.stale_data_days:
                    stale_periods.append({
                        'start_idx': start_idx,
                        'end_idx': i - 1,
                        'days': consecutive_count,
                        'start_date': df.index[start_idx],
                        'end_date': df.index[i - 1]
                    })
                consecutive_count = 0
                start_idx = None

        # Check final period
        if consecutive_count >= self.stale_data_days:
            stale_periods.append({
                'start_idx': start_idx,
                'end_idx': len(df) - 1,
                'days': consecutive_count,
                'start_date': df.index[start_idx],
                'end_date': df.index[-1]
            })

        if stale_periods:
            all_rows = []
            for period in stale_periods:
                all_rows.extend(range(period['start_idx'], period['end_idx'] + 1))

            issues.append(ValidationIssue(
                issue_type="stale_data",
                severity=ValidationSeverity.WARNING,
                message=f"Found {len(stale_periods)} periods with no price changes >= {self.stale_data_days} days",
                affected_rows=all_rows,
                details={
                    "periods": [
                        {
                            "from": p['start_date'].isoformat() if hasattr(p['start_date'], 'isoformat') else str(p['start_date']),
                            "to": p['end_date'].isoformat() if hasattr(p['end_date'], 'isoformat') else str(p['end_date']),
                            "days": p['days']
                        }
                        for p in stale_periods
                    ]
                }
            ))

        logger.debug(f"Stale data detection found {len(issues)} issues")
        return issues

    def detect_duplicates(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """
        Detect duplicate timestamps in the data.

        Args:
            df: DataFrame with datetime index

        Returns:
            List of ValidationIssue objects for duplicates
        """
        issues = []

        if isinstance(df.index, pd.DatetimeIndex):
            duplicates = df.index[df.index.duplicated()]
            if len(duplicates) > 0:
                issues.append(ValidationIssue(
                    issue_type="duplicate_timestamps",
                    severity=ValidationSeverity.ERROR,
                    message=f"Found {len(duplicates)} duplicate timestamps",
                    affected_rows=df.index.get_indexer(duplicates).tolist(),
                    details={
                        "duplicate_dates": [d.isoformat() for d in duplicates[:10]]
                    }
                ))

        logger.debug(f"Duplicate detection found {len(issues)} issues")
        return issues

    def generate_quality_report(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> QualityReport:
        """
        Generate a comprehensive data quality report.

        Runs all validation checks and compiles results into a detailed report.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock/asset symbol for identification

        Returns:
            QualityReport object with all findings
        """
        report = QualityReport(symbol=symbol)

        if df is None or len(df) == 0:
            report.issues.append(ValidationIssue(
                issue_type="empty_data",
                severity=ValidationSeverity.CRITICAL,
                message="No data provided for quality analysis"
            ))
            return report

        report.total_rows = len(df)

        # Set date range
        if isinstance(df.index, pd.DatetimeIndex):
            report.date_range_start = df.index.min()
            report.date_range_end = df.index.max()

            # Calculate expected trading days
            total_days = (report.date_range_end - report.date_range_start).days
            # Approximate: ~252 trading days per 365 calendar days
            report.trading_days_expected = int(total_days * 252 / 365)
            report.trading_days_actual = len(df)

        # Run all validation checks
        report.issues.extend(self.validate_ohlcv(df))
        report.issues.extend(self.detect_bad_ticks(df))
        report.issues.extend(self.detect_outliers(df))
        report.issues.extend(self.detect_gaps(df))
        report.issues.extend(self.detect_stale_data(df))
        report.issues.extend(self.detect_duplicates(df))

        # Calculate completeness score
        # Based on: (1 - missing_data_pct) * (1 - error_pct)
        total_cells = len(df) * len(self.REQUIRED_COLUMNS)
        missing_cells = sum(df[col].isna().sum() for col in self.REQUIRED_COLUMNS if col in df.columns)
        missing_pct = missing_cells / total_cells if total_cells > 0 else 1.0

        error_count = sum(1 for i in report.issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
        error_factor = max(0, 1 - (error_count * 0.1))  # Each error reduces score by 10%

        report.completeness_score = (1 - missing_pct) * error_factor

        # Generate summary statistics
        report.summary = {
            "Total Issues": len(report.issues),
            "Critical Issues": sum(1 for i in report.issues if i.severity == ValidationSeverity.CRITICAL),
            "Error Issues": sum(1 for i in report.issues if i.severity == ValidationSeverity.ERROR),
            "Warning Issues": sum(1 for i in report.issues if i.severity == ValidationSeverity.WARNING),
            "Info Issues": sum(1 for i in report.issues if i.severity == ValidationSeverity.INFO),
            "Missing Data %": f"{missing_pct:.2%}",
            "Data Quality Grade": self._calculate_grade(report.completeness_score)
        }

        logger.info(f"Quality report generated for {symbol}: {len(report.issues)} issues found")
        return report

    def _calculate_grade(self, score: float) -> str:
        """Convert completeness score to letter grade"""
        if score >= 0.95:
            return "A"
        elif score >= 0.90:
            return "B"
        elif score >= 0.80:
            return "C"
        elif score >= 0.70:
            return "D"
        else:
            return "F"


class DataCleaner:
    """
    Cleans and normalizes OHLCV data issues.

    Provides methods to:
    - Forward/backward fill missing prices
    - Remove or cap extreme outliers
    - Interpolate missing volume
    - Align timestamps and handle timezone issues

    Example:
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_all(df)
    """

    def __init__(
        self,
        ffill_limit: int = 5,
        outlier_cap_std: float = 3.0,
        interpolate_method: str = 'linear'
    ):
        """
        Initialize DataCleaner with configurable parameters.

        Args:
            ffill_limit: Maximum number of consecutive NaN values to forward fill
            outlier_cap_std: Standard deviations for capping outliers
            interpolate_method: Method for interpolation ('linear', 'polynomial', etc.)
        """
        self.ffill_limit = ffill_limit
        self.outlier_cap_std = outlier_cap_std
        self.interpolate_method = interpolate_method

    def forward_fill_prices(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Fill missing prices using forward fill with configurable limit.

        Args:
            df: DataFrame with OHLCV data
            columns: Columns to fill (default: OHLC columns)
            limit: Maximum consecutive fills (uses self.ffill_limit if None)

        Returns:
            DataFrame with filled values
        """
        df = df.copy()
        limit = limit if limit is not None else self.ffill_limit

        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close']

        for col in columns:
            if col in df.columns:
                original_nan = df[col].isna().sum()
                df[col] = df[col].ffill(limit=limit)
                filled = original_nan - df[col].isna().sum()
                logger.debug(f"Forward filled {filled} values in '{col}'")

        return df

    def backward_fill_prices(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        Fill missing prices using backward fill with configurable limit.

        Useful for filling initial NaN values not covered by forward fill.

        Args:
            df: DataFrame with OHLCV data
            columns: Columns to fill (default: OHLC columns)
            limit: Maximum consecutive fills

        Returns:
            DataFrame with filled values
        """
        df = df.copy()
        limit = limit if limit is not None else self.ffill_limit

        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close']

        for col in columns:
            if col in df.columns:
                original_nan = df[col].isna().sum()
                df[col] = df[col].bfill(limit=limit)
                filled = original_nan - df[col].isna().sum()
                logger.debug(f"Backward filled {filled} values in '{col}'")

        return df

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        method: Literal['drop', 'cap', 'nan'] = 'cap'
    ) -> pd.DataFrame:
        """
        Handle extreme outlier values.

        Args:
            df: DataFrame with OHLCV data
            columns: Columns to check (default: Close and Volume)
            method: How to handle outliers:
                - 'drop': Remove rows with outliers
                - 'cap': Cap values at threshold
                - 'nan': Replace outliers with NaN

        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()

        if columns is None:
            columns = ['Close', 'Volume']

        for col in columns:
            if col not in df.columns:
                continue

            mean = df[col].mean()
            std = df[col].std()

            if std == 0 or pd.isna(std):
                continue

            lower_bound = mean - (self.outlier_cap_std * std)
            upper_bound = mean + (self.outlier_cap_std * std)

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()

            if outlier_count == 0:
                continue

            if method == 'drop':
                df = df[~outlier_mask]
                logger.info(f"Dropped {outlier_count} outlier rows in '{col}'")

            elif method == 'cap':
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                logger.info(f"Capped {outlier_count} outliers in '{col}' to [{lower_bound:.2f}, {upper_bound:.2f}]")

            elif method == 'nan':
                df.loc[outlier_mask, col] = np.nan
                logger.info(f"Set {outlier_count} outliers to NaN in '{col}'")

        return df

    def interpolate_volume(
        self,
        df: pd.DataFrame,
        method: str = None
    ) -> pd.DataFrame:
        """
        Fill missing volume using interpolation.

        Args:
            df: DataFrame with OHLCV data
            method: Interpolation method (uses self.interpolate_method if None)

        Returns:
            DataFrame with interpolated volume
        """
        df = df.copy()
        method = method if method is not None else self.interpolate_method

        if 'Volume' not in df.columns:
            logger.warning("No 'Volume' column found for interpolation")
            return df

        original_nan = df['Volume'].isna().sum()
        zero_count = (df['Volume'] == 0).sum()

        # Replace zeros with NaN for interpolation
        df.loc[df['Volume'] == 0, 'Volume'] = np.nan

        # Interpolate
        df['Volume'] = df['Volume'].interpolate(method=method)

        # Fill any remaining NaN at edges
        df['Volume'] = df['Volume'].ffill().bfill()

        filled = (original_nan + zero_count) - df['Volume'].isna().sum()
        logger.info(f"Interpolated {filled} volume values using '{method}' method")

        return df

    def align_timestamps(
        self,
        df: pd.DataFrame,
        timezone: str = 'UTC',
        normalize_time: bool = True
    ) -> pd.DataFrame:
        """
        Handle timezone issues and normalize timestamps.

        Args:
            df: DataFrame with datetime index
            timezone: Target timezone (default: UTC)
            normalize_time: If True, remove time component (keep only date)

        Returns:
            DataFrame with aligned timestamps
        """
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
                logger.info("Converted index to DatetimeIndex")
            except Exception as e:
                logger.error(f"Failed to convert index to datetime: {e}")
                return df

        # Handle timezone
        if df.index.tz is None:
            # Assume UTC if no timezone
            df.index = df.index.tz_localize(timezone)
            logger.debug(f"Localized timezone to {timezone}")
        else:
            df.index = df.index.tz_convert(timezone)
            logger.debug(f"Converted timezone to {timezone}")

        # Normalize time (keep only date)
        if normalize_time:
            df.index = df.index.normalize()
            logger.debug("Normalized timestamps to date only")

        # Sort index
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            logger.debug("Sorted index chronologically")

        return df

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        keep: Literal['first', 'last'] = 'last'
    ) -> pd.DataFrame:
        """
        Remove duplicate timestamps from data.

        Args:
            df: DataFrame with datetime index
            keep: Which duplicate to keep ('first' or 'last')

        Returns:
            DataFrame with duplicates removed
        """
        df = df.copy()

        duplicate_count = df.index.duplicated().sum()
        if duplicate_count > 0:
            df = df[~df.index.duplicated(keep=keep)]
            logger.info(f"Removed {duplicate_count} duplicate timestamps (kept '{keep}')")

        return df

    def fix_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix invalid OHLC relationships (High < Low, etc.).

        Attempts to fix by using the proper min/max values.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with fixed OHLC relationships
        """
        df = df.copy()

        # For each row, ensure proper OHLC relationships
        # High should be max of O, H, L, C
        # Low should be min of O, H, L, C

        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in ohlc_cols):
            logger.warning("Cannot fix OHLC relationships: missing required columns")
            return df

        # Count violations before fix
        violations = (
            (df['High'] < df['Low']).sum() +
            (df['High'] < df['Open']).sum() +
            (df['High'] < df['Close']).sum() +
            (df['Low'] > df['Open']).sum() +
            (df['Low'] > df['Close']).sum()
        )

        if violations == 0:
            logger.debug("No OHLC violations to fix")
            return df

        # Fix by recalculating High and Low
        df['High'] = df[ohlc_cols].max(axis=1)
        df['Low'] = df[ohlc_cols].min(axis=1)

        logger.info(f"Fixed {violations} OHLC relationship violations")
        return df

    def clean_all(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        align_timestamps: bool = True,
        fill_prices: bool = True,
        interpolate_volume: bool = True,
        fix_ohlc: bool = True,
        cap_outliers: bool = False
    ) -> pd.DataFrame:
        """
        Apply all cleaning operations in recommended order.

        Args:
            df: DataFrame with OHLCV data
            remove_duplicates: Remove duplicate timestamps
            align_timestamps: Normalize timestamps
            fill_prices: Forward/backward fill missing prices
            interpolate_volume: Interpolate missing volume
            fix_ohlc: Fix OHLC relationship violations
            cap_outliers: Cap extreme outliers

        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting data cleaning pipeline (rows: {len(df)})")

        if remove_duplicates:
            df = self.remove_duplicates(df)

        if align_timestamps:
            df = self.align_timestamps(df)

        if fill_prices:
            df = self.forward_fill_prices(df)
            df = self.backward_fill_prices(df)

        if interpolate_volume:
            df = self.interpolate_volume(df)

        if fix_ohlc:
            df = self.fix_ohlc_relationships(df)

        if cap_outliers:
            df = self.remove_outliers(df, method='cap')

        logger.info(f"Data cleaning complete (rows: {len(df)})")
        return df


class MissingDataHandler:
    """
    Configurable handler for missing data with multiple strategies.

    Provides:
    - Multiple fill strategies (ffill, bfill, interpolate, drop, mean, median)
    - Gap flagging for large missing periods
    - Completeness scoring

    Example:
        handler = MissingDataHandler(strategy=MissingDataStrategy.FORWARD_FILL)
        df = handler.handle_missing(df)
        score = handler.get_completeness_score(df)
    """

    def __init__(
        self,
        strategy: Union[MissingDataStrategy, str] = MissingDataStrategy.FORWARD_FILL,
        fill_limit: int = 5,
        large_gap_threshold: int = 10
    ):
        """
        Initialize MissingDataHandler.

        Args:
            strategy: Default strategy for handling missing data
            fill_limit: Maximum consecutive fills for ffill/bfill
            large_gap_threshold: Days threshold for flagging large gaps
        """
        if isinstance(strategy, str):
            strategy = MissingDataStrategy(strategy)
        self.strategy = strategy
        self.fill_limit = fill_limit
        self.large_gap_threshold = large_gap_threshold

    def handle_missing(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        strategy: Union[MissingDataStrategy, str] = None
    ) -> pd.DataFrame:
        """
        Handle missing data using configured strategy.

        Args:
            df: DataFrame with potentially missing data
            columns: Columns to process (default: all numeric columns)
            strategy: Override default strategy for this call

        Returns:
            DataFrame with missing data handled
        """
        df = df.copy()

        if strategy is None:
            strategy = self.strategy
        elif isinstance(strategy, str):
            strategy = MissingDataStrategy(strategy)

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            original_nan = df[col].isna().sum()
            if original_nan == 0:
                continue

            if strategy == MissingDataStrategy.FORWARD_FILL:
                df[col] = df[col].ffill(limit=self.fill_limit)

            elif strategy == MissingDataStrategy.BACKWARD_FILL:
                df[col] = df[col].bfill(limit=self.fill_limit)

            elif strategy == MissingDataStrategy.INTERPOLATE:
                df[col] = df[col].interpolate(method='linear', limit=self.fill_limit)

            elif strategy == MissingDataStrategy.DROP:
                # Mark for dropping but don't drop yet (to preserve alignment)
                pass  # Will be handled after loop

            elif strategy == MissingDataStrategy.MEAN:
                df[col] = df[col].fillna(df[col].mean())

            elif strategy == MissingDataStrategy.MEDIAN:
                df[col] = df[col].fillna(df[col].median())

            elif strategy == MissingDataStrategy.ZERO:
                df[col] = df[col].fillna(0)

            filled = original_nan - df[col].isna().sum()
            logger.debug(f"Handled {filled}/{original_nan} missing values in '{col}' using {strategy.value}")

        # Handle DROP strategy
        if strategy == MissingDataStrategy.DROP:
            original_len = len(df)
            df = df.dropna(subset=columns)
            dropped = original_len - len(df)
            logger.info(f"Dropped {dropped} rows with missing data")

        return df

    def flag_large_gaps(
        self,
        df: pd.DataFrame,
        threshold_days: int = None
    ) -> pd.DataFrame:
        """
        Flag rows that follow large data gaps.

        Adds a 'large_gap_flag' column to identify data points following
        significant gaps in the time series.

        Args:
            df: DataFrame with datetime index
            threshold_days: Days threshold (uses self.large_gap_threshold if None)

        Returns:
            DataFrame with 'large_gap_flag' column added
        """
        df = df.copy()
        threshold = threshold_days if threshold_days is not None else self.large_gap_threshold

        df['large_gap_flag'] = False

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Cannot flag gaps: index is not DatetimeIndex")
            return df

        if len(df) < 2:
            return df

        # Calculate gaps
        date_diffs = df.index.to_series().diff()

        # Flag rows following large gaps
        large_gap_mask = date_diffs > pd.Timedelta(days=threshold)
        df.loc[large_gap_mask, 'large_gap_flag'] = True

        flagged_count = large_gap_mask.sum()
        if flagged_count > 0:
            logger.info(f"Flagged {flagged_count} rows following gaps > {threshold} days")

        return df

    def get_completeness_score(
        self,
        df: pd.DataFrame,
        columns: List[str] = None
    ) -> float:
        """
        Calculate data completeness as percentage of non-missing values.

        Args:
            df: DataFrame to analyze
            columns: Columns to check (default: all columns)

        Returns:
            Completeness score from 0.0 to 1.0
        """
        if df is None or len(df) == 0:
            return 0.0

        if columns is None:
            columns = df.columns.tolist()
        else:
            columns = [c for c in columns if c in df.columns]

        if len(columns) == 0:
            return 1.0

        total_cells = len(df) * len(columns)
        missing_cells = sum(df[col].isna().sum() for col in columns)

        score = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0

        logger.debug(f"Completeness score: {score:.2%} ({total_cells - missing_cells}/{total_cells} cells)")
        return score

    def get_missing_summary(
        self,
        df: pd.DataFrame,
        columns: List[str] = None
    ) -> Dict:
        """
        Get detailed summary of missing data by column.

        Args:
            df: DataFrame to analyze
            columns: Columns to check (default: all columns)

        Returns:
            Dictionary with missing data statistics per column
        """
        if df is None or len(df) == 0:
            return {"error": "Empty or None DataFrame"}

        if columns is None:
            columns = df.columns.tolist()

        summary = {
            "total_rows": len(df),
            "columns": {}
        }

        for col in columns:
            if col not in df.columns:
                continue

            missing = df[col].isna().sum()
            summary["columns"][col] = {
                "missing_count": int(missing),
                "missing_pct": float(missing / len(df)) if len(df) > 0 else 0.0,
                "dtype": str(df[col].dtype)
            }

        # Overall statistics
        total_cells = len(df) * len(columns)
        total_missing = sum(s["missing_count"] for s in summary["columns"].values())
        summary["overall_completeness"] = 1.0 - (total_missing / total_cells) if total_cells > 0 else 0.0

        return summary


# Convenience functions for quick validation
def validate_ohlcv_data(
    df: pd.DataFrame,
    symbol: str = "UNKNOWN"
) -> QualityReport:
    """
    Quick validation of OHLCV data with default settings.

    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol identifier

    Returns:
        QualityReport with all validation results
    """
    validator = DataValidator()
    return validator.generate_quality_report(df, symbol)


def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick cleaning of OHLCV data with default settings.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner()
    return cleaner.clean_all(df)


def get_data_quality_score(df: pd.DataFrame) -> float:
    """
    Get quick completeness score for data.

    Args:
        df: DataFrame to analyze

    Returns:
        Completeness score from 0.0 to 1.0
    """
    handler = MissingDataHandler()
    return handler.get_completeness_score(df)
