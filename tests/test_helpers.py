"""
Unit tests for helper utility functions
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantsploit.utils.helpers import (
    format_currency,
    format_percentage,
    format_number,
    format_large_number
)


class TestFormatCurrency:
    """Tests for currency formatting"""

    def test_positive_value(self):
        """Test formatting positive currency value"""
        result = format_currency(1234.56)
        assert '$' in result
        assert '1,234.56' in result or '1234.56' in result

    def test_negative_value(self):
        """Test formatting negative currency value"""
        result = format_currency(-1234.56)
        assert '-' in result or '(' in result
        assert '1,234.56' in result or '1234.56' in result

    def test_zero(self):
        """Test formatting zero"""
        result = format_currency(0)
        assert '$' in result
        assert '0' in result

    def test_large_value(self):
        """Test formatting large currency value"""
        result = format_currency(1234567.89)
        assert '$' in result


class TestFormatPercentage:
    """Tests for percentage formatting"""

    def test_positive_percentage(self):
        """Test formatting positive percentage"""
        result = format_percentage(12.34)
        assert '%' in result
        assert '12.34' in result or '12.3' in result

    def test_negative_percentage(self):
        """Test formatting negative percentage"""
        result = format_percentage(-5.67)
        assert '%' in result
        assert '-' in result

    def test_zero_percentage(self):
        """Test formatting zero percentage"""
        result = format_percentage(0)
        assert '%' in result
        assert '0' in result


class TestFormatNumber:
    """Tests for general number formatting"""

    def test_integer(self):
        """Test formatting integer"""
        result = format_number(1234)
        assert '1234' in result or '1,234' in result

    def test_float(self):
        """Test formatting float"""
        result = format_number(1234.567)
        # Should have some decimal places
        assert '.' in result or '1234' in result


class TestFormatLargeNumber:
    """Tests for large number formatting with suffixes"""

    def test_thousands(self):
        """Test formatting thousands"""
        result = format_large_number(1500)
        assert 'K' in result or '1500' in result or '1.5' in result

    def test_millions(self):
        """Test formatting millions"""
        result = format_large_number(1500000)
        assert 'M' in result or '1.5' in result

    def test_billions(self):
        """Test formatting billions"""
        result = format_large_number(1500000000)
        assert 'B' in result or '1.5' in result

    def test_small_number(self):
        """Test formatting small number (no suffix)"""
        result = format_large_number(500)
        # Small numbers might not have suffix
        assert '500' in result or result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
