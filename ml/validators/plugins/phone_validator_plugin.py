"""
Phone Validator Plugin

Adapter that wraps the existing PhoneValidator class to conform to the
BaseValidator interface, making it compatible with the plugin architecture.
"""

from typing import Any, List, Optional
from ml.base_validator import BaseValidator, ValidationResult
from ml.core.phone_validator import PhoneValidator


class PhoneValidatorPlugin(BaseValidator):
    """
    Phone number validator plugin that wraps the existing PhoneValidator.

    This adapter allows the existing phone validator to work with the
    new plugin-based architecture.
    """

    def __init__(self, model_path: Optional[str] = 'ml/models/phone_validator.pkl'):
        """
        Initialize the phone validator plugin.

        Args:
            model_path: Path to the trained model file
        """
        super().__init__(model_path)
        self._validator = PhoneValidator(model_path)
        self.is_trained = self._validator.is_model_loaded()

    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a single phone number.

        Args:
            value: Phone number string to validate

        Returns:
            ValidationResult with validation outcome

        Raises:
            ValueError: If model is not loaded
        """
        if not self.is_model_loaded():
            raise ValueError("Phone validation model is not loaded")

        phone_str = str(value) if value is not None else ""

        try:
            is_valid, confidence = self._validator.validate_phone(phone_str)

            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                error_type="format_error" if not is_valid else None,
                metadata={'original_value': phone_str}
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                error_type="validation_error",
                metadata={'error': str(e), 'original_value': phone_str}
            )

    def validate_batch(self, values: List[Any]) -> List[ValidationResult]:
        """
        Validate multiple phone numbers at once.

        Args:
            values: List of phone numbers to validate

        Returns:
            List of ValidationResult objects

        Raises:
            ValueError: If model is not loaded
        """
        if not self.is_model_loaded():
            raise ValueError("Phone validation model is not loaded")

        phone_strs = [str(val) if val is not None else "" for val in values]

        try:
            results = self._validator.validate_phone_batch(phone_strs)

            validation_results = []
            for phone_str, (is_valid, confidence) in zip(phone_strs, results):
                validation_results.append(
                    ValidationResult(
                        is_valid=is_valid,
                        confidence=confidence,
                        error_type="format_error" if not is_valid else None,
                        metadata={'original_value': phone_str}
                    )
                )

            return validation_results
        except Exception as e:
            # Return error results for all values
            return [
                ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    error_type="validation_error",
                    metadata={'error': str(e), 'original_value': str(val)}
                )
                for val in values
            ]

    def get_data_type(self) -> str:
        """
        Get the data type this validator handles.

        Returns:
            'phone'
        """
        return 'phone'

    def is_model_loaded(self) -> bool:
        """
        Check if the phone validation model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._validator.is_model_loaded()

    def load_model(self, filepath: str) -> bool:
        """
        Load a pre-trained phone validation model.

        Args:
            filepath: Path to the model file

        Returns:
            True if successful, False otherwise
        """
        success = self._validator.load_model(filepath)
        if success:
            self.model_path = filepath
            self.is_trained = True
        return success

    def get_model_info(self) -> dict:
        """
        Get information about the phone validation model.

        Returns:
            Dictionary with model metadata
        """
        return {
            'data_type': self.get_data_type(),
            'is_loaded': self.is_model_loaded(),
            'model_path': self.model_path,
            'model_type': 'Logistic Regression',
            'description': 'ML-based phone number validation using logistic regression'
        }
