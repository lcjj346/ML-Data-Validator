"""
PHONE VALIDATOR - Use trained ML models to validate phone numbers

BEFORE USING THIS FILE, YOU MUST TRAIN A MODEL FIRST:
---------------------------------------------------------
cd ml
python phone_model_trainer.py

HOW TO USE THE VALIDATOR:

Method 1: Quick validation of single phone numbers
---------------------------------------------------------
from phone_validator import validate_single_phone

is_valid, confidence = validate_single_phone("+1234567890")
print(f"Valid: {is_valid}, Confidence: {confidence*100:.1f}%")

Method 2: Validate multiple phone numbers at once
---------------------------------------------------------
from phone_validator import validate_phone_list

phones = ["+1234567890", "invalid_phone", "+6591234567"]
results = validate_phone_list(phones)

for phone, (is_valid, confidence) in zip(phones, results):
    status = "✅ Valid" if is_valid else "❌ Invalid" 
    print(f"{phone:<15} -> {status} ({confidence*100:.1f}% confidence)")

Method 3: Using the PhoneValidator class directly
---------------------------------------------------------
from phone_validator import PhoneValidator

validator = PhoneValidator()
if validator.is_model_loaded():
    is_valid, confidence = validator.validate_phone("+1234567890")
    print(f"Phone is {'valid' if is_valid else 'invalid'} with {confidence*100:.1f}% confidence")
else:
    print("No trained model found! Run phone_model_trainer.py first")

INTEGRATION WITH YOUR APP:
---------------------------------------------------------
# In your main application, you can use:
from ml.phone_validator import PhoneValidator

validator = PhoneValidator()
def check_phone(phone_number):
    if validator.is_model_loaded():
        is_valid, confidence = validator.validate_phone(phone_number)
        return is_valid and confidence > 0.8  # 80% confidence threshold
    else:
        return False  # Fallback if no model available

MODEL LOCATION:
---------------------------------------------------------
Expected model file: ml/trained_models/phone_validator_model.pkl
This file is created when you run the trainer.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import re
import joblib
import os


class PhoneValidator:
    """
    Phone number validator using trained ML model.
    This class is focused purely on validation - no training functionality.
    """
    
    def __init__(self, model_path='ml/trained_models/phone_validator_model.pkl'):
        self.model = None
        self.is_trained = False
        self.model_path = model_path
        
        # Try to load pre-trained model
        if os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_features(self, phone_numbers):
        """Extract features from phone numbers for ML validation"""
        features = []
        
        for phone in phone_numbers:
            phone_str = str(phone) if phone is not None else ""
            
            # Basic features for validation
            feature_dict = {
                'length': len(phone_str),
                'starts_with_plus': int(phone_str.startswith('+')),
                'digit_count': len(re.findall(r'\d', phone_str)),
                'non_digit_count': len(re.findall(r'\D', phone_str)),
                'has_spaces': int(' ' in phone_str),
                'has_dashes': int('-' in phone_str),
                'has_parentheses': int('(' in phone_str or ')' in phone_str),
                'has_letters': int(bool(re.search(r'[a-zA-Z]', phone_str))),
                'consecutive_digits': self._count_consecutive_digits(phone_str),
                'valid_length': int(8 <= len(re.findall(r'\d', phone_str)) <= 15),
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _count_consecutive_digits(self, phone_str):
        """Count maximum consecutive digits"""
        matches = re.findall(r'\d+', phone_str)
        if matches:
            return max(len(match) for match in matches)
        return 0

    def validate_phone(self, phone_number):
        """
        Validate a single phone number.
        Returns: (is_valid: bool, confidence: float)
        """
        if not self.is_trained:
            raise ValueError("Model must be loaded before validation. Train the model first.")
        
        predictions, probabilities = self.predict([phone_number])
        is_valid = bool(predictions[0])
        confidence = float(max(probabilities[0]))
        
        return is_valid, confidence
    
    def validate_phone_batch(self, phone_numbers):
        """
        Validate multiple phone numbers at once.
        Returns: list of tuples [(is_valid: bool, confidence: float), ...]
        """
        if not self.is_trained:
            raise ValueError("Model must be loaded before validation. Train the model first.")
        
        predictions, probabilities = self.predict(phone_numbers)
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            is_valid = bool(pred)
            confidence = float(max(prob))
            results.append((is_valid, confidence))
        
        return results
    
    def predict(self, phone_numbers):
        """
        Internal prediction method.
        Returns: (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be loaded before making predictions")
        
        if isinstance(phone_numbers, str):
            phone_numbers = [phone_numbers]
        
        X = self.extract_features(phone_numbers)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False
        
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.is_trained = model_data['is_trained']
            print(f"Phone validation model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def is_model_loaded(self):
        """Check if a model is loaded and ready for validation"""
        return self.is_trained and self.model is not None


# Convenience functions for easy usage
def validate_single_phone(phone_number, model_path='ml/trained_models/phone_validator_model.pkl'):
    """
    Validate a single phone number using the default model.
    Returns: (is_valid: bool, confidence: float)
    """
    validator = PhoneValidator(model_path)
    if not validator.is_model_loaded():
        raise ValueError("No trained model available. Please train the model first.")
    
    return validator.validate_phone(phone_number)


def validate_phone_list(phone_numbers, model_path='ml/trained_models/phone_validator_model.pkl'):
    """
    Validate a list of phone numbers using the default model.
    Returns: list of tuples [(is_valid: bool, confidence: float), ...]
    """
    validator = PhoneValidator(model_path)
    if not validator.is_model_loaded():
        raise ValueError("No trained model available. Please train the model first.")
    
    return validator.validate_phone_batch(phone_numbers)


if __name__ == "__main__":
    # Test the validator if model exists
    test_phones = [
        "+1234567890",     # Valid
        "+65 9123 4567",   # Valid with spaces
        "invalid_phone",   # Invalid
        "123",             # Too short
        "+1234567890123456", # Too long
    ]
    
    try:
        print("Testing Phone Validator...")
        validator = PhoneValidator()
        
        if validator.is_model_loaded():
            results = validator.validate_phone_batch(test_phones)
            
            print("\nValidation Results:")
            for phone, (is_valid, confidence) in zip(test_phones, results):
                status = "Valid" if is_valid else "Invalid"
                print(f"{phone:<20} -> {status} (confidence: {confidence*100:.1f}%)")
        else:
            print("No trained model found. Please run the trainer first.")
            
    except Exception as e:
        print(f"Error: {e}")