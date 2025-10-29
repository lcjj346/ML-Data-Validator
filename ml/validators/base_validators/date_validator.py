"""
Date Validator

Pattern-based date validator that handles multiple date formats.
Validates and parses dates in various formats.

Example usage:
    from ml.validators.date_validator import DateValidator

    validator = DateValidator()
    result = validator.validate('2024-01-15')
    print(result.is_valid, result.confidence)
"""

from typing import Any, List, Optional
import pandas as pd
from datetime import datetime
from ml.base_validator import BaseValidator, ValidationResult


class DateValidator(BaseValidator):
    """
    Date validator using pandas date parsing.

    Validates dates using:
    - Pandas date parsing (supports many formats)
    - Range validation (reasonable date ranges)
    - Format detection

    This is a rule-based validator (no ML training required).
    """

    def __init__(self,
                 min_date: Optional[str] = '1900-01-01',
                 max_date: Optional[str] = '2100-12-31',
                 model_path: Optional[str] = None):
        """
        Initialize the date validator.

        Args:
            min_date: Minimum valid date (ISO format)
            max_date: Maximum valid date (ISO format)
            model_path: Not used (rule-based validator)
        """
        super().__init__(model_path)
        self.min_date = pd.to_datetime(min_date) if min_date else None
        self.max_date = pd.to_datetime(max_date) if max_date else None
        self.is_trained = True  # Always ready

    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a single date value.

        Args:
            value: Date value to validate (string or datetime)

        Returns:
            ValidationResult with validation outcome
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return ValidationResult(
                is_valid=False,
                confidence=1.0,
                error_type="empty_value",
                metadata={'original_value': str(value)}
            )

        # Try to parse the date
        try:
            # Use pandas for flexible parsing
            parsed_date = pd.to_datetime(value, errors='raise')

            # Check if result is NaT (Not a Time)
            if pd.isna(parsed_date):
                return ValidationResult(
                    is_valid=False,
                    confidence=0.9,
                    error_type="parse_error",
                    metadata={'original_value': str(value), 'reason': 'unparseable'}
                )

            # Validate range
            if self.min_date and parsed_date < self.min_date:
                return ValidationResult(
                    is_valid=False,
                    confidence=0.8,
                    error_type="range_error",
                    metadata={
                        'original_value': str(value),
                        'parsed_date': str(parsed_date),
                        'reason': f'before_minimum_date_{self.min_date.date()}'
                    }
                )

            if self.max_date and parsed_date > self.max_date:
                return ValidationResult(
                    is_valid=False,
                    confidence=0.8,
                    error_type="range_error",
                    metadata={
                        'original_value': str(value),
                        'parsed_date': str(parsed_date),
                        'reason': f'after_maximum_date_{self.max_date.date()}'
                    }
                )

            # Determine confidence based on format clarity
            confidence = self._assess_format_confidence(str(value), parsed_date)

            return ValidationResult(
                is_valid=True,
                confidence=confidence,
                error_type=None,
                metadata={
                    'original_value': str(value),
                    'parsed_date': str(parsed_date.date()),
                    'parsed_datetime': str(parsed_date)
                }
            )

        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
            return ValidationResult(
                is_valid=False,
                confidence=0.9,
                error_type="parse_error",
                metadata={
                    'original_value': str(value),
                    'reason': 'invalid_date_format',
                    'error': str(e)
                }
            )

    def _assess_format_confidence(self, original: str, parsed: pd.Timestamp) -> float:
        """
        Assess confidence based on date format clarity.

        Args:
            original: Original date string
            parsed: Parsed datetime

        Returns:
            Confidence score (0-1)
        """
        # High confidence for ISO format (YYYY-MM-DD)
        if original.strip().count('-') == 2 and len(original.strip()) >= 10:
            return 0.95

        # High confidence for clear formats
        if '/' in original and len(original.strip()) >= 8:
            return 0.85

        # Medium confidence for numeric-only formats (ambiguous)
        if original.replace('-', '').replace('/', '').isdigit():
            return 0.7

        # Default confidence
        return 0.75

    def validate_batch(self, values: List[Any]) -> List[ValidationResult]:
        """
        Validate multiple date values at once.

        Args:
            values: List of date values to validate

        Returns:
            List of ValidationResult objects
        """
        return super().validate_batch(values)

    def get_data_type(self) -> str:
        """
        Get the data type this validator handles.

        Returns:
            'date'
        """
        return 'date'

    def is_model_loaded(self) -> bool:
        """
        Check if validator is ready.

        Returns:
            Always True (rule-based validator)
        """
        return True

    def load_model(self, filepath: str) -> bool:
        """
        Load model (not applicable for rule-based validator).

        Args:
            filepath: Model file path

        Returns:
            False (not applicable)
        """
        return False

    def get_model_info(self) -> dict:
        """
        Get information about the date validator.

        Returns:
            Dictionary with validator metadata
        """
        return {
            'data_type': self.get_data_type(),
            'is_loaded': True,
            'model_type': 'Pattern-based',
            'description': 'Multi-format date validation using pandas',
            'min_date': str(self.min_date.date()) if self.min_date else None,
            'max_date': str(self.max_date.date()) if self.max_date else None
        }


if __name__ == "__main__":
    # Test the date validator
    print("Testing Date Validator")
    print("=" * 60)

    validator = DateValidator()

    test_dates = [
        '2024-01-15',
        '01/15/2024',
        '15-01-2024',
        'January 15, 2024',
        '2024-13-45',  # Invalid
        'not a date',
        '1850-01-01',  # Too old
        '2150-01-01',  # Too far in future
        '',
    ]

    for date_str in test_dates:
        result = validator.validate(date_str)
        print(f"Date: {date_str:<20} -> Valid: {result.is_valid}, "
              f"Confidence: {result.confidence:.2f}, Error: {result.error_type}")
        if result.is_valid and result.metadata:
            print(f"      Parsed as: {result.metadata.get('parsed_date')}")
