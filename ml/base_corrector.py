"""
Base Corrector Interface

This module defines the abstract base class for all data correctors.
All domain-specific correctors (phone, email, numeric, etc.) must inherit from this class.

Example usage:
    class PhoneCorrector(BaseCorrector):
        def correct(self, value: str) -> CorrectionResult:
            # Implementation here
            pass
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
from dataclasses import dataclass


@dataclass
class CorrectionResult:
    """
    Result of a correction operation.

    Attributes:
        original_value: The original invalid value
        corrected_value: The corrected value (None if correction failed)
        confidence: Float between 0-1 indicating correction confidence
        correction_type: Type of correction applied (e.g., "typo_fix", "format_fix", "range_adjustment")
        metadata: Optional dict with additional information
    """
    original_value: Any
    corrected_value: Optional[Any]
    confidence: float
    correction_type: Optional[str] = None
    metadata: Optional[dict] = None

    def was_corrected(self) -> bool:
        """Check if correction was successful."""
        return self.corrected_value is not None and self.corrected_value != self.original_value


class BaseCorrector(ABC):
    """
    Abstract base class for all data correctors.

    All correctors must implement:
    - correct(): Single value correction
    - correct_batch(): Batch correction (with default implementation)
    - get_data_type(): Return the data type this corrector handles
    - is_model_loaded(): Check if model is ready

    Correctors should be ML-first (no rule-based fallbacks).
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the corrector.

        Args:
            model_path: Optional path to pre-trained model
        """
        self.model_path = model_path
        self.is_trained = False

    @abstractmethod
    def correct(self, value: Any) -> CorrectionResult:
        """
        Correct a single invalid value.

        Args:
            value: The invalid value to correct

        Returns:
            CorrectionResult with original, corrected value, and confidence

        Raises:
            ValueError: If model is not loaded
        """
        pass

    def correct_batch(self, values: List[Any]) -> List[CorrectionResult]:
        """
        Correct multiple values at once (optimized batch processing).

        Default implementation calls correct() for each value.
        Subclasses can override for true batch optimization.

        Args:
            values: List of invalid values to correct

        Returns:
            List of CorrectionResult objects

        Raises:
            ValueError: If model is not loaded
        """
        return [self.correct(value) for value in values]

    @abstractmethod
    def get_data_type(self) -> str:
        """
        Get the data type this corrector handles.

        Returns:
            String identifier (e.g., "phone", "email", "numeric_range", "date")
        """
        pass

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """
        Check if the model is loaded and ready for correction.

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
