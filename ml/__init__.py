"""
Simple ML Data Validator Package

A pure ML approach for data validation:
- Train on YOUR data
- Validate ANY column type
- No pre-trained models
- No complex registry
"""

from ml.feature_extractor import GenericFeatureExtractor
from ml.validator import GenericMLValidator
from ml.corrector import GenericMLCorrector

__all__ = [
    'GenericFeatureExtractor',
    'GenericMLValidator',
    'GenericMLCorrector',
]
