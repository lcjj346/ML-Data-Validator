import pandas as pd
import re
from typing import Any, Optional, Dict

class DataCorrector:
    """Rule-based data correction suggestions."""
    
    @staticmethod
    def suggest_phone_correction(phone_value: Any) -> Optional[str]:
        """Suggest corrections for phone numbers."""
        if pd.isna(phone_value):
            return None
        
        phone_str = str(phone_value).strip()
        
        # Remove all non-digit characters except +
        cleaned = re.sub(r'[^\d+]', '', phone_str)
        
        # Fix multiple + signs
        if cleaned.count('+') > 1:
            cleaned = '+' + cleaned.replace('+', '')
        
        # Remove letters and special characters
        if re.search(r'[a-zA-Z]', phone_str):
            # Try to extract just the digits
            digits_only = re.sub(r'[^\d]', '', phone_str)
            if len(digits_only) >= 8:
                # Add country code if missing
                if len(digits_only) == 8:
                    cleaned = '+65' + digits_only
                elif len(digits_only) == 10:
                    cleaned = '+' + digits_only
                else:
                    cleaned = '+' + digits_only
        
        # Ensure proper format
        if cleaned and not cleaned.startswith('+'):
            if len(cleaned) == 8:
                cleaned = '+65' + cleaned
            elif len(cleaned) >= 10:
                cleaned = '+' + cleaned
        
        return cleaned if cleaned != phone_str else None
    
    @staticmethod
    def suggest_blood_sugar_correction(blood_sugar_value: Any) -> Optional[float]:
        """Suggest corrections for blood sugar values."""
        if pd.isna(blood_sugar_value):
            return None
        
        value_str = str(blood_sugar_value).strip()
        
        # Try to extract numeric value
        try:
            value = float(value_str)
            
            # If negative, make positive
            if value < 0:
                return abs(value)
            
            # If extremely high, might be a typo (e.g., 9999 -> 99.9)
            if value > 500:
                if value >= 1000:
                    # Try dividing by 10 or 100
                    corrected = value / 100
                    if 50 <= corrected <= 200:
                        return corrected
                    corrected = value / 10
                    if 50 <= corrected <= 500:
                        return corrected
                
                # If still too high, suggest a reasonable range
                return 120.0  # Average normal value
            
            return None  # Value is already reasonable
            
        except ValueError:
            # Try to extract numbers from string
            numbers = re.findall(r'\d+\.?\d*', value_str)
            if numbers:
                try:
                    suggested_value = float(numbers[0])
                    if 50 <= suggested_value <= 500:
                        return suggested_value
                except ValueError:
                    pass
            
            return 100.0  # Default reasonable value
    
    @staticmethod
    def suggest_general_correction(value: Any, column_name: str) -> Optional[str]:
        """General correction suggestions based on common patterns."""
        if pd.isna(value):
            return None
        
        value_str = str(value).strip()
        
        # If it's clearly meant to be numeric but has text
        if re.search(r'\d', value_str) and re.search(r'[a-zA-Z]', value_str):
            # Extract just the numbers
            numbers = re.findall(r'\d+\.?\d*', value_str)
            if numbers:
                return numbers[0]
        
        # If it's empty or just whitespace
        if not value_str or value_str.isspace():
            return "N/A"
        
        return None