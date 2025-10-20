"""
Base Validator Interface

This module defines the abstract base class for all data validators.
All domain-specific validators (phone, email, numeric, etc.) must inherit from this class.

Example usage:
    class PhoneValidator(BaseValidator):
        def validate(self, value: str) -> ValidationResult:
            # Implementation here
            pass
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class ValidationResult:
    """
    Result of a validation operation.

    Attributes:
        is_valid: Boolean indicating if the value is valid
        confidence: Float between 0-1 indicating model confidence
        error_type: Optional string describing the type of error (e.g., "format_error", "range_error")
        metadata: Optional dict with additional information
    """
    is_valid: bool
    confidence: float
    error_type: Optional[str] = None
    metadata: Optional[dict] = None


class BaseValidator(ABC):
    """
    Abstract base class for all data validators.

    All validators must implement:
    - validate(): Single value validation
    - validate_batch(): Batch validation (with default implementation)
    - get_data_type(): Return the data type this validator handles
    - is_model_loaded(): Check if model is ready

    Validators should be ML-first (no rule-based fallbacks).
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the validator.

        Args:
            model_path: Optional path to pre-trained model
        """
        self.model_path = model_path
        self.is_trained = False

    @abstractmethod
    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a single value.

        Args:
            value: The value to validate

        Returns:
            ValidationResult with is_valid, confidence, and optional metadata

        Raises:
            ValueError: If model is not loaded
        """
        pass

    def validate_batch(self, values: List[Any]) -> List[ValidationResult]:
        """
        Validate multiple values at once (optimized batch processing).

        Default implementation calls validate() for each value.
        Subclasses can override for true batch optimization.

        Args:
            values: List of values to validate

        Returns:
            List of ValidationResult objects

        Raises:
            ValueError: If model is not loaded
        """
        return [self.validate(value) for value in values]

    @abstractmethod
    def get_data_type(self) -> str:
        """
        Get the data type this validator handles.

        Returns:
            String identifier (e.g., "phone", "email", "numeric_range", "date")
        """
        pass

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """
        Check if the model is loaded and ready for validation.

        Returns:
            True if model is loaded, False otherwise
        """
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """
        Load a pre-trained model from file.

        Args:
            filepath: Path to the saved model file

        Returns:
            True if successful, False otherwise
        """
        pass

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        return {
            'data_type': self.get_data_type(),
            'is_loaded': self.is_model_loaded(),
            'model_path': self.model_path
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data_type='{self.get_data_type()}', loaded={self.is_model_loaded()})"
