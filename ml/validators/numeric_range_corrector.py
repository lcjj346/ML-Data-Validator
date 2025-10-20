"""
Numeric Range Corrector

ML-based corrector for numeric data that falls outside expected ranges.
Uses statistical methods and ML to suggest corrections for invalid numeric values.

Example usage:
    from ml.validators.numeric_range_corrector import NumericRangeCorrector

    corrector = NumericRangeCorrector(range_type='blood_sugar')
    result = corrector.correct(3000)  # Likely a typo: 300.0
    print(result.corrected_value)
"""

from typing import Any, List, Optional
import numpy as np
from ml.base_corrector import BaseCorrector, CorrectionResult
from ml.validators.numeric_range_validator import NumericRangeValidator


class NumericRangeCorrector(BaseCorrector):
    """
    Corrector for numeric values that fall outside expected ranges.

    Uses heuristics to suggest corrections:
    - Decimal point errors (3000 -> 300.0)
    - Unit conversion errors (180 cm -> 1.80 m)
    - Magnitude errors (1200 -> 120)
    - Typos and digit errors

    Attributes:
        range_type: Type of numeric data
        validator: NumericRangeValidator instance for validation
    """

    def __init__(self,
                 range_type: str = 'generic',
                 expected_min: Optional[float] = None,
                 expected_max: Optional[float] = None,
                 model_path: Optional[str] = None):
        """
        Initialize the numeric range corrector.

        Args:
            range_type: Type of numeric data (e.g., 'blood_sugar', 'height')
            expected_min: Custom minimum value
            expected_max: Custom maximum value
            model_path: Path to pre-trained model (not used yet, for future ML enhancements)
        """
        super().__init__(model_path)
        self.range_type = range_type
        self.validator = NumericRangeValidator(range_type, expected_min, expected_max)
        self.is_trained = True  # Rule-based, always ready

    def correct(self, value: Any) -> CorrectionResult:
        """
        Correct an invalid numeric value.

        Args:
            value: Invalid numeric value to correct

        Returns:
            CorrectionResult with correction outcome
        """
        # Try to convert to float
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return CorrectionResult(
                original_value=value,
                corrected_value=None,
                confidence=0.0,
                correction_type=None,
                metadata={'reason': 'not_numeric'}
            )

        # Check if value is already valid
        validation_result = self.validator.validate(num_value)
        if validation_result.is_valid:
            return CorrectionResult(
                original_value=num_value,
                corrected_value=None,
                confidence=validation_result.confidence,
                correction_type=None,
                metadata={'reason': 'already_valid'}
            )

        # Try various correction strategies
        corrections = []

        # Strategy 1: Decimal point shift
        for shift in [-3, -2, -1, 1, 2, 3]:
            candidate = num_value / (10 ** abs(shift)) if shift < 0 else num_value * (10 ** shift)
            result = self.validator.validate(candidate)
            if result.is_valid:
                corrections.append({
                    'value': candidate,
                    'confidence': result.confidence * 0.7,  # Reduce confidence for inferred corrections
                    'type': 'decimal_shift',
                    'description': f'Shifted decimal point by {shift} places'
                })

        # Strategy 2: Remove/add zeros
        str_value = str(int(num_value)) if num_value == int(num_value) else str(num_value)

        # Remove trailing zeros
        if str_value.endswith('0'):
            candidate = num_value / 10
            result = self.validator.validate(candidate)
            if result.is_valid:
                corrections.append({
                    'value': candidate,
                    'confidence': result.confidence * 0.8,
                    'type': 'remove_zero',
                    'description': 'Removed trailing zero'
                })

        # Strategy 3: Divide by common factors (for unit conversion errors)
        for factor in [100, 10, 2]:
            candidate = num_value / factor
            result = self.validator.validate(candidate)
            if result.is_valid:
                corrections.append({
                    'value': candidate,
                    'confidence': result.confidence * 0.6,
                    'type': 'unit_conversion',
                    'description': f'Divided by {factor} (possible unit error)'
                })

        # Strategy 4: Multiply by common factors
        for factor in [2, 10, 100]:
            candidate = num_value * factor
            result = self.validator.validate(candidate)
            if result.is_valid:
                corrections.append({
                    'value': candidate,
                    'confidence': result.confidence * 0.6,
                    'type': 'unit_conversion',
                    'description': f'Multiplied by {factor} (possible unit error)'
                })

        # Strategy 5: Clamp to expected range (last resort, low confidence)
        if num_value < self.validator.expected_min:
            candidate = self.validator.expected_min
        elif num_value > self.validator.expected_max:
            candidate = self.validator.expected_max
        else:
            candidate = None

        if candidate is not None:
            corrections.append({
                'value': candidate,
                'confidence': 0.3,  # Low confidence for clamping
                'type': 'range_clamp',
                'description': f'Clamped to expected range'
            })

        # Select best correction
        if corrections:
            # Sort by confidence
            corrections.sort(key=lambda x: x['confidence'], reverse=True)
            best = corrections[0]

            return CorrectionResult(
                original_value=num_value,
                corrected_value=best['value'],
                confidence=best['confidence'],
                correction_type=best['type'],
                metadata={
                    'description': best['description'],
                    'alternative_corrections': len(corrections) - 1
                }
            )
        else:
            # No valid correction found
            return CorrectionResult(
                original_value=num_value,
                corrected_value=None,
                confidence=0.0,
                correction_type=None,
                metadata={'reason': 'no_valid_correction'}
            )

    def correct_batch(self, values: List[Any]) -> List[CorrectionResult]:
        """
        Correct multiple invalid numeric values at once.

        Args:
            values: List of invalid numeric values to correct

        Returns:
            List of CorrectionResult objects
        """
        return super().correct_batch(values)

    def get_data_type(self) -> str:
        """
        Get the data type this corrector handles.

        Returns:
            f'numeric_range:{range_type}'
        """
        return f'numeric_range:{self.range_type}'

    def is_model_loaded(self) -> bool:
        """
        Check if the corrector is ready.

        Returns:
            Always True (rule-based corrector)
        """
        return True

    def load_model(self, filepath: str) -> bool:
        """
        Load a pre-trained model (not implemented for rule-based corrector).

        Args:
            filepath: Path to the model file

        Returns:
            False (not applicable for rule-based)
        """
        # Rule-based corrector doesn't need model loading
        return False

    def get_model_info(self) -> dict:
        """
        Get information about the numeric range corrector.

        Returns:
            Dictionary with corrector metadata
        """
        return {
            'data_type': self.get_data_type(),
            'is_loaded': True,
            'model_type': 'Rule-based',
            'range_type': self.range_type,
            'description': f'Rule-based numeric correction for {self.range_type}',
            'strategies': [
                'decimal_shift',
                'remove_trailing_zeros',
                'unit_conversion',
                'range_clamp'
            ]
        }


if __name__ == "__main__":
    # Test the numeric range corrector
    print("Testing Numeric Range Corrector")
    print("=" * 60)

    # Test blood sugar corrector
    print("\n1. Blood Sugar Corrector")
    print("-" * 60)
    corrector = NumericRangeCorrector(range_type='blood_sugar')

    test_values = [3000, 1200, 8, 500, 10, 1.5]

    for value in test_values:
        result = corrector.correct(value)
        if result.was_corrected():
            print(f"Value: {value:<10} -> Corrected: {result.corrected_value:.1f}, "
                  f"Confidence: {result.confidence:.2f}, "
                  f"Type: {result.correction_type}")
        else:
            print(f"Value: {value:<10} -> No correction possible")

    print("\n2. Height Corrector (cm)")
    print("-" * 60)
    corrector_height = NumericRangeCorrector(range_type='height')

    test_heights = [1800, 1.75, 17500, 5, 250]

    for value in test_heights:
        result = corrector_height.correct(value)
        if result.was_corrected():
            print(f"Height: {value:<10} -> Corrected: {result.corrected_value:.1f} cm, "
                  f"Confidence: {result.confidence:.2f}")
        else:
            print(f"Height: {value:<10} -> No correction possible")
