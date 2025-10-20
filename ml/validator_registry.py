"""
Validator Registry System

This module provides a centralized registry for all data validators and correctors.
It allows the system to dynamically discover and use appropriate validators/correctors
based on column types or explicit registration.

Example usage:
    from ml.validator_registry import ValidatorRegistry

    # Register a validator
    registry = ValidatorRegistry()
    registry.register_validator('phone', phone_validator_instance)

    # Get validator for a data type
    validator = registry.get_validator('phone')

    # Auto-detect and validate
    result = registry.validate_value('phone', '+1234567890')
"""

from typing import Dict, Optional, Any, List
from ml.base_validator import BaseValidator, ValidationResult
from ml.base_corrector import BaseCorrector, CorrectionResult


class ValidatorRegistry:
    """
    Central registry for all validators and correctors.

    This class manages the mapping between data types and their corresponding
    validator/corrector implementations. It provides a single point of access
    for validation and correction operations.

    Attributes:
        _validators: Dictionary mapping data type to validator instance
        _correctors: Dictionary mapping data type to corrector instance
    """

    def __init__(self):
        """Initialize empty registry."""
        self._validators: Dict[str, BaseValidator] = {}
        self._correctors: Dict[str, BaseCorrector] = {}

    def register_validator(self, data_type: str, validator: BaseValidator) -> None:
        """
        Register a validator for a specific data type.

        Args:
            data_type: The data type identifier (e.g., 'phone', 'email')
            validator: The validator instance

        Raises:
            TypeError: If validator doesn't inherit from BaseValidator
        """
        if not isinstance(validator, BaseValidator):
            raise TypeError(f"Validator must inherit from BaseValidator, got {type(validator)}")

        self._validators[data_type] = validator
        print(f"Registered validator for '{data_type}': {validator.__class__.__name__}")

    def register_corrector(self, data_type: str, corrector: BaseCorrector) -> None:
        """
        Register a corrector for a specific data type.

        Args:
            data_type: The data type identifier (e.g., 'phone', 'email')
            corrector: The corrector instance

        Raises:
            TypeError: If corrector doesn't inherit from BaseCorrector
        """
        if not isinstance(corrector, BaseCorrector):
            raise TypeError(f"Corrector must inherit from BaseCorrector, got {type(corrector)}")

        self._correctors[data_type] = corrector
        print(f"Registered corrector for '{data_type}': {corrector.__class__.__name__}")

    def get_validator(self, data_type: str) -> Optional[BaseValidator]:
        """
        Get the validator for a specific data type.

        Args:
            data_type: The data type identifier

        Returns:
            Validator instance or None if not registered
        """
        return self._validators.get(data_type)

    def get_corrector(self, data_type: str) -> Optional[BaseCorrector]:
        """
        Get the corrector for a specific data type.

        Args:
            data_type: The data type identifier

        Returns:
            Corrector instance or None if not registered
        """
        return self._correctors.get(data_type)

    def has_validator(self, data_type: str) -> bool:
        """
        Check if a validator is registered for the data type.

        Args:
            data_type: The data type identifier

        Returns:
            True if validator exists and is loaded, False otherwise
        """
        validator = self._validators.get(data_type)
        return validator is not None and validator.is_model_loaded()

    def has_corrector(self, data_type: str) -> bool:
        """
        Check if a corrector is registered for the data type.

        Args:
            data_type: The data type identifier

        Returns:
            True if corrector exists and is loaded, False otherwise
        """
        corrector = self._correctors.get(data_type)
        return corrector is not None and corrector.is_model_loaded()

    def validate_value(self, data_type: str, value: Any) -> Optional[ValidationResult]:
        """
        Validate a single value using the appropriate validator.

        Args:
            data_type: The data type identifier
            value: The value to validate

        Returns:
            ValidationResult or None if no validator available
        """
        validator = self.get_validator(data_type)
        if validator and validator.is_model_loaded():
            return validator.validate(value)
        return None

    def validate_batch(self, data_type: str, values: List[Any]) -> Optional[List[ValidationResult]]:
        """
        Validate multiple values using the appropriate validator.

        Args:
            data_type: The data type identifier
            values: List of values to validate

        Returns:
            List of ValidationResult or None if no validator available
        """
        validator = self.get_validator(data_type)
        if validator and validator.is_model_loaded():
            return validator.validate_batch(values)
        return None

    def correct_value(self, data_type: str, value: Any) -> Optional[CorrectionResult]:
        """
        Correct a single value using the appropriate corrector.

        Args:
            data_type: The data type identifier
            value: The value to correct

        Returns:
            CorrectionResult or None if no corrector available
        """
        corrector = self.get_corrector(data_type)
        if corrector and corrector.is_model_loaded():
            return corrector.correct(value)
        return None

    def correct_batch(self, data_type: str, values: List[Any]) -> Optional[List[CorrectionResult]]:
        """
        Correct multiple values using the appropriate corrector.

        Args:
            data_type: The data type identifier
            values: List of values to correct

        Returns:
            List of CorrectionResult or None if no corrector available
        """
        corrector = self.get_corrector(data_type)
        if corrector and corrector.is_model_loaded():
            return corrector.correct_batch(values)
        return None

    def get_supported_types(self) -> Dict[str, dict]:
        """
        Get information about all supported data types.

        Returns:
            Dictionary mapping data type to capabilities dict
        """
        supported = {}

        all_types = set(self._validators.keys()) | set(self._correctors.keys())

        for data_type in all_types:
            supported[data_type] = {
                'has_validator': self.has_validator(data_type),
                'has_corrector': self.has_corrector(data_type),
                'validator_info': self._validators[data_type].get_model_info() if data_type in self._validators else None,
                'corrector_info': self._correctors[data_type].get_model_info() if data_type in self._correctors else None
            }

        return supported

    def unregister_validator(self, data_type: str) -> bool:
        """
        Remove a validator from the registry.

        Args:
            data_type: The data type identifier

        Returns:
            True if validator was removed, False if not found
        """
        if data_type in self._validators:
            del self._validators[data_type]
            return True
        return False

    def unregister_corrector(self, data_type: str) -> bool:
        """
        Remove a corrector from the registry.

        Args:
            data_type: The data type identifier

        Returns:
            True if corrector was removed, False if not found
        """
        if data_type in self._correctors:
            del self._correctors[data_type]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered validators and correctors."""
        self._validators.clear()
        self._correctors.clear()

    def __repr__(self) -> str:
        validator_types = list(self._validators.keys())
        corrector_types = list(self._correctors.keys())
        return (f"ValidatorRegistry(validators={validator_types}, "
                f"correctors={corrector_types})")


# Global registry instance
_global_registry: Optional[ValidatorRegistry] = None


def get_global_registry() -> ValidatorRegistry:
    """
    Get the global validator registry instance.

    Returns:
        Global ValidatorRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ValidatorRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _global_registry
    _global_registry = None
