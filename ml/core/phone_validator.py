"""
Phone Validator - Use trained ML models to validate phone numbers.

This module provides phone number validation using a trained Logistic Regression model.
Features are extracted using the centralized PhoneFeatureExtractor class.

Example usage:
    from ml.validator import PhoneValidator

    validator = PhoneValidator('saved_models/phone_validator_model.pkl')
    if validator.is_model_loaded():
        is_valid, confidence = validator.validate_phone("+1234567890")
        print(f"Valid: {is_valid}, Confidence: {confidence*100:.1f}%")
"""

import os
from typing import List, Tuple, Optional
import joblib
from sklearn.linear_model import LogisticRegression
from ml.core.phone_features import PhoneFeatureExtractor


class PhoneValidator:
    """
    Phone number validator using trained Logistic Regression model.

    This class focuses purely on validation - no training functionality.
    Features are extracted using PhoneFeatureExtractor.

    Attributes:
        model: Trained LogisticRegression model instance
        is_trained: Boolean flag indicating if model is loaded
        model_path: Path to the model file
    """

    def __init__(self, model_path: str = '../saved_models/phone_validator_model.pkl'):
        """
        Initialize PhoneValidator and optionally load pre-trained model.

        Args:
            model_path: Path to saved model file (.pkl)
        """
        self.model: Optional[LogisticRegression] = None
        self.is_trained: bool = False
        self.model_path: str = model_path

        # Try to load pre-trained model
        if os.path.exists(model_path):
            self.load_model(model_path)

    def validate_phone(self, phone_number: str) -> Tuple[bool, float]:
        """
        Validate a single phone number.

        Args:
            phone_number: Phone number string to validate

        Returns:
            Tuple of (is_valid, confidence) where:
                - is_valid: Boolean indicating if phone is valid
                - confidence: Float between 0-1 indicating model confidence

        Raises:
            ValueError: If model is not loaded
        """
        if not self.is_trained:
            raise ValueError("Model must be loaded before validation. Train the model first.")

        predictions, probabilities = self._predict([phone_number])
        is_valid = bool(predictions[0])
        confidence = float(max(probabilities[0]))

        return is_valid, confidence

    def validate_phone_batch(self, phone_numbers: List[str]) -> List[Tuple[bool, float]]:
        """
        Validate multiple phone numbers at once (optimized batch processing).

        Args:
            phone_numbers: List of phone number strings

        Returns:
            List of tuples [(is_valid, confidence), ...] for each phone

        Raises:
            ValueError: If model is not loaded
        """
        if not self.is_trained:
            raise ValueError("Model must be loaded before validation. Train the model first.")

        predictions, probabilities = self._predict(phone_numbers)

        results = []
        for pred, prob in zip(predictions, probabilities):
            is_valid = bool(pred)
            confidence = float(max(prob))
            results.append((is_valid, confidence))

        return results

    def _predict(self, phone_numbers: List[str]) -> Tuple:
        """
        Internal prediction method.

        Args:
            phone_numbers: List of phone numbers (or single phone as list)

        Returns:
            Tuple of (predictions, probabilities)

        Raises:
            ValueError: If model is not loaded
        """
        if not self.is_trained:
            raise ValueError("Model must be loaded before making predictions")

        if isinstance(phone_numbers, str):
            phone_numbers = [phone_numbers]

        # Use centralized feature extractor
        X = PhoneFeatureExtractor.extract_features(phone_numbers)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return predictions, probabilities

    def load_model(self, filepath: str) -> bool:
        """
        Load a pre-trained model from file.

        Args:
            filepath: Path to saved model (.pkl file)

        Returns:
            True if model loaded successfully, False otherwise
        """
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

    def is_model_loaded(self) -> bool:
        """
        Check if a model is loaded and ready for validation.

        Returns:
            True if model is loaded and trained, False otherwise
        """
        return self.is_trained and self.model is not None


# Convenience functions for easy usage
def validate_single_phone(
    phone_number: str,
    model_path: str = '../saved_models/phone_validator_model.pkl'
) -> Tuple[bool, float]:
    """
    Validate a single phone number using the default model.

    Convenience function for quick validation without creating a validator instance.

    Args:
        phone_number: Phone number string to validate
        model_path: Path to saved model file

    Returns:
        Tuple of (is_valid, confidence)

    Raises:
        ValueError: If no trained model is available
    """
    validator = PhoneValidator(model_path)
    if not validator.is_model_loaded():
        raise ValueError("No trained model available. Please train the model first.")

    return validator.validate_phone(phone_number)


def validate_phone_list(
    phone_numbers: List[str],
    model_path: str = '../saved_models/phone_validator_model.pkl'
) -> List[Tuple[bool, float]]:
    """
    Validate a list of phone numbers using the default model.

    Convenience function for batch validation without creating a validator instance.

    Args:
        phone_numbers: List of phone number strings
        model_path: Path to saved model file

    Returns:
        List of tuples [(is_valid, confidence), ...]

    Raises:
        ValueError: If no trained model is available
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