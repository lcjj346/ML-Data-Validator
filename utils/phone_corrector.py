"""
Phone Correction Utilities

This module provides utility functions for generating intelligent phone number
correction suggestions using ML models with rule-based fallbacks.
"""

from typing import Optional
import pandas as pd


def generate_phone_suggestion(invalid_phone: str, edit_distance_corrector=None) -> str:
    """
    Generate intelligent suggestions for invalid phone numbers.

    Uses XGBoost ML model when available, falls back to rule-based approach otherwise.

    Args:
        invalid_phone: Invalid phone number string to correct
        edit_distance_corrector: Optional EditDistanceCorrector instance

    Returns:
        Corrected phone number string, or empty string if cannot correct
    """
    if not invalid_phone or pd.isna(invalid_phone):
        return ""

    phone_str = str(invalid_phone).strip()

    # PRIORITY 1: Try XGBoost Edit Distance Corrector (ML-based)
    if edit_distance_corrector and hasattr(edit_distance_corrector, 'model') and edit_distance_corrector.model:
        try:
            corrected = edit_distance_corrector.correct_phone(phone_str)
            # XGBoost returns None if it can't correct
            if corrected and corrected != phone_str:
                return corrected
        except Exception:
            # If XGBoost fails, fall through to rule-based approach
            pass

    # FALLBACK: Rule-based approach (simple corrections)
    return _rule_based_correction(phone_str)


def _rule_based_correction(phone_str: str) -> str:
    """
    Apply rule-based corrections to phone number.

    Args:
        phone_str: Phone number string to correct

    Returns:
        Corrected phone number string, or empty string if cannot correct
    """
    # Replace common letter typos with similar-looking digits
    corrections = {
        'o': '0', 'O': '0',  # o to 0
        'l': '1', 'L': '1', 'I': '1', 'i': '1',  # l/I to 1
        'e': '3', 'E': '3',  # e to 3
        's': '5', 'S': '5',  # s to 5
        'b': '8', 'B': '8',  # b to 8
        'g': '9', 'G': '9',  # g to 9
        'a': '2', 'A': '2',  # a to 2
    }

    suggestion = phone_str
    for letter, digit in corrections.items():
        suggestion = suggestion.replace(letter, digit)

    # Remove spaces, dashes, dots, parentheses
    suggestion = suggestion.replace(' ', '').replace('-', '').replace('.', '').replace('(', '').replace(')', '')

    # Remove any remaining non-digit characters except +
    cleaned = ''
    for char in suggestion:
        if char.isdigit() or char == '+':
            cleaned += char
    suggestion = cleaned

    # Ensure it starts with +
    if suggestion and not suggestion.startswith('+'):
        suggestion = _add_country_code(suggestion)

    # Validate final result
    if not _is_valid_phone_format(suggestion):
        return ""

    return suggestion


def _add_country_code(phone: str) -> str:
    """
    Add appropriate country code to phone number.

    Args:
        phone: Phone number without country code

    Returns:
        Phone number with country code
    """
    digit_count = len([c for c in phone if c.isdigit()])

    if digit_count == 10:
        return '+1' + phone  # Assume US
    elif digit_count == 11 and phone.startswith('1'):
        return '+' + phone  # US with country code
    elif digit_count >= 8:
        return '+' + phone  # International

    return phone


def _is_valid_phone_format(phone: str) -> bool:
    """
    Validate phone number format.

    Args:
        phone: Phone number to validate

    Returns:
        True if format is valid, False otherwise
    """
    if not phone or len(phone) < 8:
        return False

    if phone.count('+') > 1:
        return False

    digit_count = sum(c.isdigit() for c in phone)
    if digit_count < 7 or digit_count > 15:
        return False

    return True
