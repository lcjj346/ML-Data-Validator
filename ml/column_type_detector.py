"""
Column Type Detector

This module provides automatic detection of column data types from CSV/Excel files.
It uses heuristics and pattern matching to identify common data types like phone numbers,
emails, dates, numeric ranges, etc.

Example usage:
    from ml.column_type_detector import ColumnTypeDetector

    detector = ColumnTypeDetector()
    data_type = detector.detect_column_type('PhoneNumber', df['PhoneNumber'])
    # Returns: 'phone'
"""

import re
import pandas as pd
from typing import Optional, List, Dict
from collections import Counter


class ColumnTypeDetector:
    """
    Automatically detect data types from column names and values.

    This class uses a combination of:
    - Column name pattern matching
    - Value pattern analysis
    - Statistical analysis of column values

    Supported data types:
    - phone: Phone numbers
    - email: Email addresses
    - date: Date values
    - numeric_range: Numeric values with expected ranges (blood sugar, height, weight, etc.)
    - text: Generic text data
    - categorical: Limited set of categorical values
    """

    def __init__(self):
        """Initialize the column type detector with pattern matchers."""
        # Column name patterns (case-insensitive)
        self.name_patterns = {
            'phone': [r'phone', r'mobile', r'tel', r'contact'],
            'email': [r'email', r'e-mail', r'mail'],
            'date': [r'date', r'time', r'created', r'updated', r'dob', r'birth'],
            'numeric_range': {
                'blood_sugar': [r'glucose', r'sugar', r'bg', r'blood.*sugar'],
                'height': [r'height', r'tall'],
                'weight': [r'weight', r'mass', r'kg', r'lbs'],
                'age': [r'age', r'years.*old'],
                'temperature': [r'temp', r'fever'],
                'blood_pressure': [r'bp', r'pressure', r'systolic', r'diastolic'],
                'calories': [r'calor', r'kcal', r'energy'],
                'heart_rate': [r'heart.*rate', r'hr', r'pulse', r'bpm'],
                'steps': [r'steps', r'step.*count'],
            }
        }

        # Value patterns
        self.value_patterns = {
            'phone': r'^\+?[\d\s\-\(\)\.]{7,20}$',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        }

    def detect_column_type(self, column_name: str, column_data: pd.Series, sample_size: int = 100) -> str:
        """
        Detect the data type of a column based on its name and values.

        Args:
            column_name: The name of the column
            column_data: Pandas Series containing the column values
            sample_size: Number of samples to analyze (default: 100)

        Returns:
            String identifier of detected data type
        """
        # Step 1: Try to detect from column name
        name_based_type = self._detect_from_name(column_name)
        if name_based_type:
            # Verify with sample data
            if self._verify_type_with_data(name_based_type, column_data, sample_size):
                return name_based_type

        # Step 2: Detect from data patterns
        data_based_type = self._detect_from_data(column_data, sample_size)
        if data_based_type:
            return data_based_type

        # Step 3: Default fallback
        return self._detect_generic_type(column_data, sample_size)

    def _detect_from_name(self, column_name: str) -> Optional[str]:
        """
        Detect data type from column name using pattern matching.

        Args:
            column_name: The name of the column

        Returns:
            Detected data type or None
        """
        column_name_lower = column_name.lower()

        # Check simple types first
        for data_type, patterns in self.name_patterns.items():
            if data_type == 'numeric_range':
                continue  # Handle separately

            for pattern in patterns:
                if re.search(pattern, column_name_lower):
                    return data_type

        # Check numeric_range subtypes
        for subtype, patterns in self.name_patterns['numeric_range'].items():
            for pattern in patterns:
                if re.search(pattern, column_name_lower):
                    return f'numeric_range:{subtype}'

        return None

    def _detect_from_data(self, column_data: pd.Series, sample_size: int) -> Optional[str]:
        """
        Detect data type by analyzing column values.

        Args:
            column_data: Column values
            sample_size: Number of samples to check

        Returns:
            Detected data type or None
        """
        # Get non-null sample
        sample = column_data.dropna().head(sample_size)
        if len(sample) == 0:
            return None

        # Convert to strings for pattern matching
        sample_str = sample.astype(str)

        # Check phone pattern
        phone_matches = sum(1 for val in sample_str if re.match(self.value_patterns['phone'], val))
        if phone_matches / len(sample) > 0.7:  # 70% threshold
            return 'phone'

        # Check email pattern
        email_matches = sum(1 for val in sample_str if re.match(self.value_patterns['email'], val))
        if email_matches / len(sample) > 0.7:
            return 'email'

        # Check if numeric
        try:
            numeric_values = pd.to_numeric(sample, errors='coerce')
            if numeric_values.notna().sum() / len(sample) > 0.8:  # 80% are numbers
                # It's numeric, try to determine the subtype
                return self._detect_numeric_subtype(numeric_values.dropna())
        except:
            pass

        # Check if date/datetime
        try:
            # Try common date formats first to avoid warning
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
                try:
                    date_values = pd.to_datetime(sample, format=fmt, errors='coerce')
                    if date_values.notna().sum() / len(sample) > 0.7:
                        return 'date'
                except:
                    continue

            # Fallback to flexible parsing (may show warning but works)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                date_values = pd.to_datetime(sample, errors='coerce')
                if date_values.notna().sum() / len(sample) > 0.7:
                    return 'date'
        except:
            pass

        return None

    def _detect_numeric_subtype(self, numeric_data: pd.Series) -> str:
        """
        Detect specific numeric range type based on value ranges.

        Args:
            numeric_data: Numeric column values

        Returns:
            Specific numeric_range subtype
        """
        min_val = numeric_data.min()
        max_val = numeric_data.max()
        mean_val = numeric_data.mean()

        # Blood sugar (typically 50-400 mg/dL)
        if 40 <= min_val and max_val <= 500 and 60 <= mean_val <= 200:
            return 'numeric_range:blood_sugar'

        # Height in cm (typically 50-250 cm)
        if 40 <= min_val and max_val <= 300 and 100 <= mean_val <= 200:
            return 'numeric_range:height'

        # Weight in kg (typically 2-300 kg)
        if 1 <= min_val and max_val <= 400 and 20 <= mean_val <= 150:
            return 'numeric_range:weight'

        # Age (typically 0-120)
        if 0 <= min_val and max_val <= 150 and 0 <= mean_val <= 100:
            return 'numeric_range:age'

        # Temperature (typically 35-42 Celsius or 95-108 Fahrenheit)
        if ((35 <= min_val and max_val <= 43) or (95 <= min_val and max_val <= 110)):
            return 'numeric_range:temperature'

        # Generic numeric range
        return 'numeric_range:generic'

    def _verify_type_with_data(self, detected_type: str, column_data: pd.Series, sample_size: int) -> bool:
        """
        Verify that the detected type matches actual data patterns.

        Args:
            detected_type: The type detected from column name
            column_data: Column values
            sample_size: Number of samples to check

        Returns:
            True if verification passes, False otherwise
        """
        sample = column_data.dropna().head(sample_size)
        if len(sample) == 0:
            return True  # Can't verify, assume correct

        sample_str = sample.astype(str)

        # Verify phone
        if detected_type == 'phone':
            phone_matches = sum(1 for val in sample_str if re.match(self.value_patterns['phone'], val))
            return phone_matches / len(sample) > 0.5  # At least 50% match

        # Verify email
        if detected_type == 'email':
            email_matches = sum(1 for val in sample_str if re.match(self.value_patterns['email'], val))
            return email_matches / len(sample) > 0.5

        # Verify numeric_range
        if detected_type.startswith('numeric_range'):
            try:
                numeric_values = pd.to_numeric(sample, errors='coerce')
                return numeric_values.notna().sum() / len(sample) > 0.7
            except:
                return False

        # Verify date
        if detected_type == 'date':
            try:
                date_values = pd.to_datetime(sample, errors='coerce')
                return date_values.notna().sum() / len(sample) > 0.5
            except:
                return False

        return True

    def _detect_generic_type(self, column_data: pd.Series, sample_size: int) -> str:
        """
        Detect generic type when specific patterns don't match.

        Args:
            column_data: Column values
            sample_size: Number of samples to check

        Returns:
            Generic data type identifier
        """
        sample = column_data.dropna().head(sample_size)
        if len(sample) == 0:
            return 'text'

        # Check if categorical (limited unique values)
        unique_ratio = len(sample.unique()) / len(sample)
        if unique_ratio < 0.1:  # Less than 10% unique values
            return 'categorical'

        # Default to text
        return 'text'

    def detect_all_columns(self, dataframe: pd.DataFrame) -> Dict[str, str]:
        """
        Detect data types for all columns in a dataframe.

        Args:
            dataframe: Pandas DataFrame

        Returns:
            Dictionary mapping column names to detected data types
        """
        detected_types = {}

        for column_name in dataframe.columns:
            detected_type = self.detect_column_type(column_name, dataframe[column_name])
            detected_types[column_name] = detected_type

        return detected_types

    def get_column_type_summary(self, dataframe: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get a summary of detected column types grouped by type.

        Args:
            dataframe: Pandas DataFrame

        Returns:
            Dictionary mapping data types to lists of column names
        """
        detected_types = self.detect_all_columns(dataframe)

        summary = {}
        for column_name, data_type in detected_types.items():
            if data_type not in summary:
                summary[data_type] = []
            summary[data_type].append(column_name)

        return summary


if __name__ == "__main__":
    # Test the detector
    test_data = pd.DataFrame({
        'PhoneNumber': ['+1234567890', '+6591234567', '123-456-7890'],
        'Email': ['test@example.com', 'user@domain.com', 'admin@test.org'],
        'BloodSugar': [120, 95, 180],
        'Height': [170, 165, 180],
        'Name': ['Alice', 'Bob', 'Charlie']
    })

    detector = ColumnTypeDetector()
    results = detector.detect_all_columns(test_data)

    print("Column Type Detection Results:")
    print("=" * 50)
    for column, data_type in results.items():
        print(f"{column:<20} -> {data_type}")

    print("\nType Summary:")
    print("=" * 50)
    summary = detector.get_column_type_summary(test_data)
    for data_type, columns in summary.items():
        print(f"{data_type:<25} : {', '.join(columns)}")
