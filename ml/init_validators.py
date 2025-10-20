"""
Validator Initialization Module

This module initializes and registers all available validators and correctors
with the global registry. Import this at app startup to make all validators available.

Example usage:
    from ml.init_validators import initialize_validators

    registry = initialize_validators()
    # Now all validators are registered and ready to use
"""

from typing import Dict, Any
from ml.validator_registry import ValidatorRegistry, get_global_registry
from ml.validators import (
    PhoneValidatorPlugin,
    PhoneCorrectorPlugin,
    NumericRangeValidator,
    NumericRangeCorrector,
    EmailValidator,
    DateValidator
)


def initialize_validators(registry: ValidatorRegistry = None) -> ValidatorRegistry:
    """
    Initialize and register all available validators and correctors.

    Args:
        registry: Optional custom registry (uses global registry if None)

    Returns:
        ValidatorRegistry with all validators registered
    """
    if registry is None:
        registry = get_global_registry()

    print("Initializing ML Data Validator System...")
    print("=" * 60)

    # Register Phone Validator and Corrector
    try:
        phone_validator = PhoneValidatorPlugin('saved_models/phone_validator_model.pkl')
        registry.register_validator('phone', phone_validator)
        print(f"[OK] Phone Validator: {('Loaded' if phone_validator.is_model_loaded() else 'Not loaded (train model first)')}")
    except Exception as e:
        print(f"[FAIL] Phone Validator: Failed to initialize - {e}")

    try:
        phone_corrector = PhoneCorrectorPlugin('saved_models/edit_distance_corrector.pkl')
        registry.register_corrector('phone', phone_corrector)
        print(f"[OK] Phone Corrector: {('Loaded' if phone_corrector.is_model_loaded() else 'Not loaded (train model first)')}")
    except Exception as e:
        print(f"[FAIL] Phone Corrector: Failed to initialize - {e}")

    # Register Numeric Range Validators (various types)
    numeric_types = [
        ('blood_sugar', 'Blood Sugar Levels'),
        ('height', 'Height (cm)'),
        ('weight', 'Weight (kg)'),
        ('age', 'Age (years)'),
        ('temperature', 'Body Temperature'),
        ('blood_pressure_systolic', 'Systolic BP'),
        ('blood_pressure_diastolic', 'Diastolic BP'),
        ('calories', 'Calories (kcal)'),
        ('heart_rate', 'Heart Rate (bpm)'),
        ('steps', 'Daily Steps'),
    ]

    for range_type, description in numeric_types:
        try:
            validator = NumericRangeValidator(range_type=range_type)
            corrector = NumericRangeCorrector(range_type=range_type)

            registry.register_validator(f'numeric_range:{range_type}', validator)
            registry.register_corrector(f'numeric_range:{range_type}', corrector)

            print(f"[OK] Numeric Validator/Corrector ({description}): Ready")
        except Exception as e:
            print(f"[FAIL] Numeric Validator/Corrector ({description}): Failed - {e}")

    # Register Email Validator
    try:
        email_validator = EmailValidator()
        registry.register_validator('email', email_validator)
        print(f"[OK] Email Validator: Ready")
    except Exception as e:
        print(f"[FAIL] Email Validator: Failed to initialize - {e}")

    # Register Date Validator
    try:
        date_validator = DateValidator()
        registry.register_validator('date', date_validator)
        print(f"[OK] Date Validator: Ready")
    except Exception as e:
        print(f"[FAIL] Date Validator: Failed to initialize - {e}")

    print("=" * 60)
    print(f"Initialization complete! {len(registry._validators)} validators and {len(registry._correctors)} correctors registered.")
    print()

    return registry


def get_validator_status() -> Dict[str, Any]:
    """
    Get status of all registered validators and correctors.

    Returns:
        Dictionary with validator/corrector status information
    """
    registry = get_global_registry()
    supported_types = registry.get_supported_types()

    status = {
        'total_validators': sum(1 for info in supported_types.values() if info['has_validator']),
        'total_correctors': sum(1 for info in supported_types.values() if info['has_corrector']),
        'types': supported_types
    }

    return status


def print_validator_status():
    """Print a formatted status report of all validators."""
    status = get_validator_status()

    print("\nValidator System Status")
    print("=" * 60)
    print(f"Total Validators: {status['total_validators']}")
    print(f"Total Correctors: {status['total_correctors']}")
    print()

    print("Registered Data Types:")
    print("-" * 60)

    for data_type, info in sorted(status['types'].items()):
        validator_status = "[OK]" if info['has_validator'] else "[FAIL]"
        corrector_status = "[OK]" if info['has_corrector'] else "[FAIL]"

        print(f"{data_type:<35} | Validator: {validator_status} | Corrector: {corrector_status}")

    print("=" * 60)


if __name__ == "__main__":
    # Initialize validators and print status
    registry = initialize_validators()
    print_validator_status()

    # Test with some sample data
    print("\nQuick Validation Test:")
    print("=" * 60)

    test_cases = [
        ('phone', '+1234567890'),
        ('email', 'user@example.com'),
        ('numeric_range:blood_sugar', 120),
        ('numeric_range:height', 175),
        ('date', '2024-01-15'),
    ]

    for data_type, value in test_cases:
        result = registry.validate_value(data_type, value)
        if result:
            print(f"{data_type:<30} | Value: {str(value):<20} | Valid: {result.is_valid} | Confidence: {result.confidence:.2f}")
        else:
            print(f"{data_type:<30} | Value: {str(value):<20} | No validator available")

    print("=" * 60)
