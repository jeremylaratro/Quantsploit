#!/usr/bin/env python3
"""
Test script to verify quarterly period generation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from quantsploit.utils.comprehensive_backtest import (
    parse_quarters,
    get_fiscal_quarter_dates,
    find_quarter_periods,
    ComprehensiveBacktester
)
from datetime import datetime


def test_parse_quarters():
    """Test the quarter parser"""
    print("Testing quarter parser...")

    tests = [
        ('2', [2]),
        ('1,2,3', [1, 2, 3]),
        ('4', [4]),
        ('1,3', [1, 3]),
        ('2,1,3', [1, 2, 3]),  # Should be sorted and deduplicated
    ]

    for quarter_str, expected in tests:
        result = parse_quarters(quarter_str)
        status = "✓" if result == expected else "✗"
        print(f"  {status} parse_quarters('{quarter_str}') = {result} (expected {expected})")

    # Test invalid formats
    print("\nTesting invalid quarter formats...")
    invalid_tests = ['5', '0', 'abc', '1,2,5']
    for quarter_str in invalid_tests:
        try:
            parse_quarters(quarter_str)
            print(f"  ✗ parse_quarters('{quarter_str}') should have raised ValueError")
        except ValueError as e:
            print(f"  ✓ parse_quarters('{quarter_str}') correctly raised ValueError")


def test_fiscal_quarter_dates():
    """Test fiscal quarter date calculation"""
    print("\n\nTesting fiscal quarter date calculation...")

    tests = [
        (2024, 1, '2024-01-01', '2024-03-31'),
        (2024, 2, '2024-04-01', '2024-06-30'),
        (2024, 3, '2024-07-01', '2024-09-30'),
        (2024, 4, '2024-10-01', '2024-12-31'),
    ]

    for year, quarter, expected_start, expected_end in tests:
        start_date, end_date = get_fiscal_quarter_dates(year, quarter)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        status = "✓" if start_str == expected_start and end_str == expected_end else "✗"
        print(f"  {status} Q{quarter} {year}: {start_str} to {end_str}")


def test_find_quarter_periods():
    """Test quarter period finding"""
    print("\n\nTesting quarter period finding...")

    # Test single quarter with multiple periods
    print("\nTest: Last 4 Q2s")
    periods = find_quarter_periods([2], 4)
    print(f"Generated {len(periods)} periods:")
    for period in periods:
        print(f"  - {period.description}: {period.start_date} to {period.end_date}")

    # Test quarter range without period count
    print("\n\nTest: Most recent Q1, Q2, Q3")
    periods = find_quarter_periods([1, 2, 3], None)
    print(f"Generated {len(periods)} periods:")
    for period in periods:
        print(f"  - {period.description}: {period.start_date} to {period.end_date}")

    # Test single quarter without period count
    print("\n\nTest: Most recent Q4")
    periods = find_quarter_periods([4], None)
    print(f"Generated {len(periods)} periods:")
    for period in periods:
        print(f"  - {period.description}: {period.start_date} to {period.end_date}")


def test_generate_test_periods_with_quarters():
    """Test generate_test_periods with quarter parameters"""
    print("\n\nTesting generate_test_periods with quarters...")

    backtester = ComprehensiveBacktester(symbols=['AAPL'])

    # Test with quarter parameter
    print("\nTest: --quarter 2 --period 3")
    periods = backtester.generate_test_periods(quarters='2', num_periods=3)
    print(f"Generated {len(periods)} periods:")
    for period in periods[:3]:
        print(f"  - {period.description}: {period.start_date} to {period.end_date}")

    # Test with quarter range
    print("\n\nTest: --quarter 1,2")
    periods = backtester.generate_test_periods(quarters='1,2')
    print(f"Generated {len(periods)} periods:")
    for period in periods:
        print(f"  - {period.description}: {period.start_date} to {period.end_date}")


if __name__ == '__main__':
    test_parse_quarters()
    test_fiscal_quarter_dates()
    test_find_quarter_periods()
    test_generate_test_periods_with_quarters()
    print("\n✓ All tests completed!")
