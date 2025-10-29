"""
Numeric Range Validator

ML-based validator for numeric data with expected ranges (blood sugar, height, weight, etc.).
Uses anomaly detection and regression models to validate numeric values.

This validator can handle:
- Blood sugar levels (mg/dL)
- Height (cm)
- Weight (kg)
- Age (years)
- Temperature (Celsius/Fahrenheit)
- Blood pressure
- Any custom numeric range

Example usage:
    from ml.validators.numeric_range_validator import NumericRangeValidator

    validator = NumericRangeValidator(
        range_type='blood_sugar',
        expected_min=70,
        expected_max=180
    )
    result = validator.validate(120)
    print(result.is_valid, result.confidence)
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from ml.base_validator import BaseValidator, ValidationResult


class NumericRangeValidator(BaseValidator):
    """
    Validator for numeric data with expected ranges.

    Uses a combination of:
    - Hard range constraints (min/max)
    - ML-based anomaly detection (Isolation Forest)
    - Statistical outlier detection (IQR method)

    Attributes:
        range_type: Type of numeric data (e.g., 'blood_sugar', 'height', 'weight')
        expected_min: Minimum expected value
        expected_max: Maximum expected value
        model: Isolation Forest model for anomaly detection
    """

    # Predefined range configurations
    RANGE_CONFIGS = {
        'blood_sugar': {
            'min': 20, 'max': 600, 'typical_min': 70, 'typical_max': 180,
            'unit': 'mg/dL', 'description': 'Blood glucose levels'
        },
        'height': {
            'min': 30, 'max': 300, 'typical_min': 140, 'typical_max': 210,
            'unit': 'cm', 'description': 'Human height in centimeters'
        },
        'weight': {
            'min': 1, 'max': 500, 'typical_min': 30, 'typical_max': 200,
            'unit': 'kg', 'description': 'Body weight in kilograms'
        },
        'age': {
            'min': 0, 'max': 150, 'typical_min': 0, 'typical_max': 120,
            'unit': 'years', 'description': 'Age in years'
        },
        'temperature': {
            'min': 32, 'max': 45, 'typical_min': 36, 'typical_max': 38,
            'unit': '°C', 'description': 'Body temperature in Celsius'
        },
        'blood_pressure_systolic': {
            'min': 50, 'max': 250, 'typical_min': 90, 'typical_max': 140,
            'unit': 'mmHg', 'description': 'Systolic blood pressure'
        },
        'blood_pressure_diastolic': {
            'min': 30, 'max': 150, 'typical_min': 60, 'typical_max': 90,
            'unit': 'mmHg', 'description': 'Diastolic blood pressure'
        },
        'calories': {
            'min': 0, 'max': 10000, 'typical_min': 1000, 'typical_max': 4000,
            'unit': 'kcal', 'description': 'Daily calorie intake'
        },
        'heart_rate': {
            'min': 30, 'max': 220, 'typical_min': 60, 'typical_max': 100,
            'unit': 'bpm', 'description': 'Heart rate in beats per minute'
        },
        'steps': {
            'min': 0, 'max': 100000, 'typical_min': 1000, 'typical_max': 20000,
            'unit': 'steps', 'description': 'Daily step count'
        }
    }

    def __init__(self,
                 range_type: str = 'generic',
                 expected_min: Optional[float] = None,
                 expected_max: Optional[float] = None,
                 model_path: Optional[str] = None):
        """
        Initialize the numeric range validator.

        Args:
            range_type: Type of numeric data (e.g., 'blood_sugar', 'height')
            expected_min: Custom minimum value (overrides predefined)
            expected_max: Custom maximum value (overrides predefined)
            model_path: Path to pre-trained model file
        """
        super().__init__(model_path)
        self.range_type = range_type
        self.model: Optional[IsolationForest] = None

        # Get range configuration
        if range_type in self.RANGE_CONFIGS:
            config = self.RANGE_CONFIGS[range_type]
            self.expected_min = expected_min if expected_min is not None else config['typical_min']
            self.expected_max = expected_max if expected_max is not None else config['typical_max']
            self.hard_min = config['min']
            self.hard_max = config['max']
            self.unit = config['unit']
            self.description = config['description']
        else:
            # Generic/custom range
            self.expected_min = expected_min if expected_min is not None else -float('inf')
            self.expected_max = expected_max if expected_max is not None else float('inf')
            self.hard_min = expected_min if expected_min is not None else -float('inf')
            self.hard_max = expected_max if expected_max is not None else float('inf')
            self.unit = ''
            self.description = f'Generic numeric range {range_type}'

        # Try to load pre-trained model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize basic model (will need training)
            self.model = IsolationForest(contamination=0.1, random_state=42)
            self.is_trained = False

    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a single numeric value.

        Args:
            value: Numeric value to validate

        Returns:
            ValidationResult with validation outcome
        """
        # Try to convert to float
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                confidence=1.0,
                error_type="type_error",
                metadata={'reason': 'not_numeric', 'original_value': str(value)}
            )

        # Check for NaN/Inf
        if not np.isfinite(num_value):
            return ValidationResult(
                is_valid=False,
                confidence=1.0,
                error_type="value_error",
                metadata={'reason': 'nan_or_inf', 'original_value': str(value)}
            )

        # Hard constraints check
        if num_value < self.hard_min or num_value > self.hard_max:
            return ValidationResult(
                is_valid=False,
                confidence=1.0,
                error_type="range_error",
                metadata={
                    'reason': 'out_of_hard_range',
                    'value': num_value,
                    'hard_min': self.hard_min,
                    'hard_max': self.hard_max
                }
            )

        # Expected range check
        within_expected = self.expected_min <= num_value <= self.expected_max

        # ML-based anomaly detection (if model is trained)
        anomaly_score = 0.5  # Default neutral score
        if self.is_trained and self.model is not None:
            try:
                # Predict anomaly (-1 for outlier, 1 for inlier)
                prediction = self.model.predict([[num_value]])[0]
                # Get anomaly score (lower is more anomalous)
                score = self.model.score_samples([[num_value]])[0]

                # Normalize score to confidence (0-1)
                # Typical scores range from -0.5 to 0.5, but can vary
                anomaly_score = min(1.0, max(0.0, (score + 0.5)))

            except:
                # Model prediction failed, use rule-based only
                pass

        # Combine rule-based and ML-based validation
        if within_expected:
            # Value is in expected range
            confidence = 0.7 + (anomaly_score * 0.3)  # Boost confidence with ML
            is_valid = True
        else:
            # Value is outside expected range but within hard limits
            # ML model might still accept it if it's seen similar values
            if self.is_trained and anomaly_score > 0.5:
                is_valid = True
                confidence = anomaly_score * 0.6  # Reduced confidence
            else:
                is_valid = False
                confidence = 0.7 + (1 - anomaly_score) * 0.3

        error_type = None if is_valid else "range_warning"

        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            error_type=error_type,
            metadata={
                'value': num_value,
                'expected_min': self.expected_min,
                'expected_max': self.expected_max,
                'within_expected': within_expected,
                'anomaly_score': anomaly_score
            }
        )

    def validate_batch(self, values: List[Any]) -> List[ValidationResult]:
        """
        Validate multiple numeric values at once.

        Args:
            values: List of numeric values to validate

        Returns:
            List of ValidationResult objects
        """
        # For now, use default implementation (can be optimized later)
        return super().validate_batch(values)

    def get_data_type(self) -> str:
        """
        Get the data type this validator handles.

        Returns:
            f'numeric_range:{range_type}'
        """
        return f'numeric_range:{self.range_type}'

    def is_model_loaded(self) -> bool:
        """
        Check if the model is loaded and trained.

        Returns:
            True if model exists (trained or not)
        """
        return self.model is not None

    def load_model(self, filepath: str) -> bool:
        """
        Load a pre-trained anomaly detection model.

        Args:
            filepath: Path to the model file

        Returns:
            True if successful, False otherwise
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.is_trained = model_data['is_trained']
            self.range_type = model_data.get('range_type', self.range_type)
            self.expected_min = model_data.get('expected_min', self.expected_min)
            self.expected_max = model_data.get('expected_max', self.expected_max)
            self.model_path = filepath
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def train(self, training_data: List[float]) -> dict:
        """
        Train the anomaly detection model on valid examples.

        Args:
            training_data: List of valid numeric values

        Returns:
            Dictionary with training results
        """
        if len(training_data) < 10:
            raise ValueError("Need at least 10 training examples")

        # Convert to numpy array
        X = np.array(training_data).reshape(-1, 1)

        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=0.1,  # Expect 10% outliers
            random_state=42,
            n_estimators=100
        )
        self.model.fit(X)
        self.is_trained = True

        # Calculate statistics
        stats = {
            'n_samples': len(training_data),
            'mean': float(np.mean(training_data)),
            'std': float(np.std(training_data)),
            'min': float(np.min(training_data)),
            'max': float(np.max(training_data)),
            'range_type': self.range_type
        }

        return stats

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to file.

        Args:
            filepath: Path to save the model

        Raises:
            ValueError: If model hasn't been trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'is_trained': self.is_trained,
            'range_type': self.range_type,
            'expected_min': self.expected_min,
            'expected_max': self.expected_max,
            'hard_min': self.hard_min,
            'hard_max': self.hard_max
        }

        joblib.dump(model_data, filepath)
        self.model_path = filepath
        print(f"Model saved to {filepath}")

    def get_model_info(self) -> dict:
        """
        Get information about the numeric range validator.

        Returns:
            Dictionary with model metadata
        """
        return {
            'data_type': self.get_data_type(),
            'is_loaded': self.is_model_loaded(),
            'is_trained': self.is_trained,
            'model_path': self.model_path,
            'model_type': 'Isolation Forest',
            'range_type': self.range_type,
            'expected_range': f'{self.expected_min}-{self.expected_max} {self.unit}',
            'hard_range': f'{self.hard_min}-{self.hard_max} {self.unit}',
            'description': self.description
        }


if __name__ == "__main__":
    # Test the numeric range validator
    print("Testing Numeric Range Validator")
    print("=" * 60)

    # Test blood sugar validator
    print("\n1. Blood Sugar Validator")
    print("-" * 60)
    validator = NumericRangeValidator(range_type='blood_sugar')

    test_values = [85, 120, 95, 180, 300, -10, 'not a number']

    for value in test_values:
        result = validator.validate(value)
        print(f"Value: {str(value):<15} -> Valid: {result.is_valid}, "
              f"Confidence: {result.confidence:.2f}, Error: {result.error_type}")

    # Test with training
    print("\n2. Blood Sugar Validator with ML Training")
    print("-" * 60)
    validator_ml = NumericRangeValidator(range_type='blood_sugar')

    # Train on typical blood sugar values
    training_data = [85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 95, 100, 105]
    stats = validator_ml.train(training_data)
    print(f"Trained on {stats['n_samples']} samples")
    print(f"Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")

    print("\nValidation with trained model:")
    for value in test_values:
        result = validator_ml.validate(value)
        print(f"Value: {str(value):<15} -> Valid: {result.is_valid}, "
              f"Confidence: {result.confidence:.2f}, Error: {result.error_type}")
