#!/usr/bin/env python3
"""
Test script to verify custom period generation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from quantsploit.utils.comprehensive_backtest import parse_time_span, ComprehensiveBacktester


def test_parse_time_span():
    """Test the time span parser"""
    print("Testing time span parser...")

    tests = [
        ('2y', 730),
        ('6m', 180),
        ('4w', 28),
        ('90d', 90),
        ('1y', 365),
        ('12m', 360),
    ]

    for time_str, expected_days in tests:
        result = parse_time_span(time_str)
        status = "✓" if result == expected_days else "✗"
        print(f"  {status} parse_time_span('{time_str}') = {result} days (expected {expected_days})")

    # Test invalid formats
    print("\nTesting invalid formats...")
    invalid_tests = ['2years', 'abc', '12', 'm6']
    for time_str in invalid_tests:
        try:
            parse_time_span(time_str)
            print(f"  ✗ parse_time_span('{time_str}') should have raised ValueError")
        except ValueError as e:
            print(f"  ✓ parse_time_span('{time_str}') correctly raised ValueError: {e}")


def test_generate_custom_periods():
    """Test custom period generation"""
    print("\n\nTesting custom period generation...")

    # Create a backtester instance
    backtester = ComprehensiveBacktester(
        symbols=['AAPL'],
        initial_capital=100000
    )

    # Test case: 4 periods of 6 months each over 2 years
    print("\nTest: 4 periods of 6 months each over 2 years")
    periods = backtester.generate_test_periods(
        tspan='2y',
        bspan='6m',
        num_periods=4
    )

    print(f"Generated {len(periods)} periods:")
    for i, period in enumerate(periods, 1):
        print(f"  {i}. {period.description}")
        print(f"     Start: {period.start_date}, End: {period.end_date}")

    # Test case: default periods (should return standard periods)
    print("\n\nTest: Default periods (no custom params)")
    default_periods = backtester.generate_test_periods()

    print(f"Generated {len(default_periods)} default periods:")
    for i, period in enumerate(default_periods[:3], 1):  # Show just first 3
        print(f"  {i}. {period.description}")
        print(f"     Start: {period.start_date}, End: {period.end_date}")
    print(f"  ... and {len(default_periods) - 3} more periods")


if __name__ == '__main__':
    test_parse_time_span()
    test_generate_custom_periods()
    print("\n✓ All tests completed!")
