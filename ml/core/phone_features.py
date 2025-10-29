"""
Base Feature Extractor for Phone Number Validation

This module provides a centralized feature extraction class to eliminate code duplication
between validator.py and model_trainer.py.
"""

import re
from typing import List, Dict
import pandas as pd


class PhoneFeatureExtractor:
    """
    Centralized feature extraction for phone number validation.

    Extracts 10 features from phone numbers:
    1. length - Total length of phone string
    2. starts_with_plus - Binary: starts with '+'
    3. digit_count - Count of digit characters
    4. non_digit_count - Count of non-digit characters
    5. has_spaces - Binary: contains spaces
    6. has_dashes - Binary: contains dashes
    7. has_parentheses - Binary: contains parentheses
    8. has_letters - Binary: contains letters
    9. consecutive_digits - Maximum consecutive digit sequence length
    10. valid_length - Binary: digit count between 8-15
    """

    @staticmethod
    def extract_features(phone_numbers: List[str]) -> pd.DataFrame:
        """
        Extract features from phone numbers for ML validation.

        Args:
            phone_numbers: List of phone number strings to extract features from

        Returns:
            DataFrame with 10 feature columns
        """
        features = []

        for phone in phone_numbers:
            phone_str = str(phone) if phone is not None else ""

            feature_dict = {
                'length': len(phone_str),
                'starts_with_plus': int(phone_str.startswith('+')),
                'digit_count': len(re.findall(r'\d', phone_str)),
                'non_digit_count': len(re.findall(r'\D', phone_str)),
                'has_spaces': int(' ' in phone_str),
                'has_dashes': int('-' in phone_str),
                'has_parentheses': int('(' in phone_str or ')' in phone_str),
                'has_letters': int(bool(re.search(r'[a-zA-Z]', phone_str))),
                'consecutive_digits': PhoneFeatureExtractor._count_consecutive_digits(phone_str),
                'valid_length': int(8 <= len(re.findall(r'\d', phone_str)) <= 15),
            }

            features.append(feature_dict)

        return pd.DataFrame(features)

    @staticmethod
    def _count_consecutive_digits(phone_str: str) -> int:
        """
        Count maximum consecutive digits in phone string.

        Args:
            phone_str: Phone number string

        Returns:
            Maximum length of consecutive digit sequence
        """
        matches = re.findall(r'\d+', phone_str)
        if matches:
            return max(len(match) for match in matches)
        return 0
