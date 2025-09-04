import pandas as pd
import numpy as np
import re
from typing import Tuple, Dict, Any

class DataPreprocessor:
    """Preprocesses data for ML model training and prediction."""
    
    def __init__(self):
        self.feature_extractors = {
            'phone': self._extract_phone_features,
            'blood_sugar': self._extract_blood_sugar_features,
            'general': self._extract_general_features
        }
    
    def _extract_phone_features(self, value: str) -> Dict[str, float]:
        """Extract features from phone number strings."""
        if pd.isna(value):
            return {
                'length': 0,
                'has_plus': 0,
                'digit_count': 0,
                'special_char_count': 0,
                'has_letters': 0,
                'consecutive_plus': 0
            }
        
        value_str = str(value).strip()
        
        return {
            'length': len(value_str),
            'has_plus': 1 if '+' in value_str else 0,
            'digit_count': sum(c.isdigit() for c in value_str),
            'special_char_count': sum(not c.isalnum() and c not in ['+', '-', ' '] for c in value_str),
            'has_letters': 1 if any(c.isalpha() for c in value_str) else 0,
            'consecutive_plus': value_str.count('++')
        }
    
    def _extract_blood_sugar_features(self, value: Any) -> Dict[str, float]:
        """Extract features from blood sugar values."""
        if pd.isna(value):
            return {
                'is_numeric': 0,
                'value': 0,
                'is_negative': 0,
                'is_extreme': 0,
                'has_letters': 0
            }
        
        value_str = str(value).strip()
        
        try:
            numeric_val = float(value_str)
            return {
                'is_numeric': 1,
                'value': numeric_val,
                'is_negative': 1 if numeric_val < 0 else 0,
                'is_extreme': 1 if numeric_val > 500 or numeric_val < 10 else 0,
                'has_letters': 0
            }
        except ValueError:
            return {
                'is_numeric': 0,
                'value': 0,
                'is_negative': 0,
                'is_extreme': 0,
                'has_letters': 1 if any(c.isalpha() for c in value_str) else 0
            }
    
    def _extract_general_features(self, value: Any) -> Dict[str, float]:
        """Extract general features for any column."""
        if pd.isna(value):
            return {
                'is_null': 1,
                'length': 0,
                'is_numeric': 0,
                'has_special_chars': 0
            }
        
        value_str = str(value).strip()
        
        try:
            float(value_str)
            is_numeric = 1
        except ValueError:
            is_numeric = 0
        
        return {
            'is_null': 0,
            'length': len(value_str),
            'is_numeric': is_numeric,
            'has_special_chars': 1 if re.search(r'[^a-zA-Z0-9\s]', value_str) else 0
        }
    
    def extract_features(self, df: pd.DataFrame, column: str, feature_type: str = 'general') -> pd.DataFrame:
        """Extract features for a specific column."""
        extractor = self.feature_extractors.get(feature_type, self.feature_extractors['general'])
        
        features_list = []
        for value in df[column]:
            features = extractor(value)
            features_list.append(features)
        
        feature_df = pd.DataFrame(features_list)
        feature_df.columns = [f"{column}_{col}" for col in feature_df.columns]
        
        return feature_df