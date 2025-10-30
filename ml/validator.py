"""
Generic ML Validator - Works for ANY data type

Train once on YOUR data, validate forever.
Works for: phone, email, address, name, or ANY custom column type.

Key: Uses YOUR training data, not pre-trained models.
"""

import os
import joblib
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ml.feature_extractor import GenericFeatureExtractor


class GenericMLValidator:
    """
    One validator for ALL data types.

    Train it on YOUR data:
    - Phone numbers → Learns phone patterns
    - Emails → Learns email patterns
    - Addresses → Learns YOUR address format
    - Names → Learns name patterns
    - Custom data → Learns whatever you teach it!
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize generic validator.

        Args:
            model_path: Path to load trained model (optional)
        """
        self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        self.feature_extractor = GenericFeatureExtractor()
        self.is_trained = False
        self.model_path = model_path
        self.data_type = "unknown"

        # Try to load if path provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, training_data: List[Tuple[str, str]], data_type: str = "custom"):
        """
        Train validator on YOUR data.

        Args:
            training_data: List of (text, label) tuples
                          label should be "valid" or "invalid"
            data_type: Name of data type (e.g., "phone", "email", "address")

        Example:
            training_data = [
                ("+1234567890", "valid"),
                ("123", "invalid"),
                ("invalid_phone", "invalid"),
            ]
            validator.train(training_data, "phone")
        """
        print(f"Training Generic ML Validator for '{data_type}'...")
        print(f"Training examples: {len(training_data)}")

        # Extract features
        X = [self.feature_extractor.extract_features(text) for text, label in training_data]
        y = [1 if label.lower() == "valid" else 0 for text, label in training_data]

        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        self.data_type = data_type

        # Calculate accuracy
        accuracy = self.model.score(X, y)
        print(f"Training accuracy: {accuracy:.2%}")
        print("Training complete!")

    def train_from_csv(self, csv_path: str, text_col: str = "text", label_col: str = "label", data_type: str = "custom"):
        """
        Train from CSV file.

        Args:
            csv_path: Path to CSV with columns: text, label
            text_col: Name of text column
            label_col: Name of label column (should contain "valid" or "invalid")
            data_type: Name of data type

        CSV Format:
            text,label
            +1234567890,valid
            123,invalid
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training file not found: {csv_path}")

        print(f"Loading training data from {csv_path}...")
        df = pd.read_csv(csv_path)

        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"CSV must have columns: {text_col}, {label_col}")

        training_data = list(zip(df[text_col], df[label_col]))
        self.train(training_data, data_type)

    def validate(self, text: str) -> Tuple[bool, float]:
        """
        Validate a single value.

        Args:
            text: Text to validate

        Returns:
            (is_valid, confidence) where:
                is_valid: True if valid, False if invalid
                confidence: 0.0-1.0 confidence score
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or load() first.")

        features = self.feature_extractor.extract_features(text)
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        confidence = float(max(probabilities))

        return bool(prediction), confidence

    def validate_batch(self, texts: List[str]) -> List[Tuple[bool, float]]:
        """
        Validate multiple values at once (faster).

        Args:
            texts: List of texts to validate

        Returns:
            List of (is_valid, confidence) tuples
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or load() first.")

        # Extract features for all texts
        X = [self.feature_extractor.extract_features(text) for text in texts]

        # Batch predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        results = []
        for pred, prob in zip(predictions, probabilities):
            is_valid = bool(pred)
            confidence = float(max(prob))
            results.append((is_valid, confidence))

        return results

    def save(self, filepath: str):
        """
        Save trained model.

        Args:
            filepath: Path to save model (e.g., "models/phone_validator.pkl")
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'data_type': self.data_type,
            'is_trained': self.is_trained,
        }

        joblib.dump(model_data, filepath)
        self.model_path = filepath
        print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> bool:
        """
        Load trained model.

        Args:
            filepath: Path to saved model

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return False

        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.data_type = model_data['data_type']
            self.is_trained = model_data['is_trained']
            self.model_path = filepath
            print(f"Model loaded: {self.data_type} validator")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_info(self) -> dict:
        """Get model information"""
        return {
            'data_type': self.data_type,
            'is_trained': self.is_trained,
            'model_path': self.model_path,
            'model_type': 'Logistic Regression',
        }


if __name__ == "__main__":
    # Example: Train a phone validator
    print("=" * 70)
    print("GENERIC ML VALIDATOR - TEST")
    print("=" * 70)

    # Create validator
    validator = GenericMLValidator()

    # Example training data (you would use YOUR data)
    training_data = [
        ("+1234567890", "valid"),
        ("+65 9123 4567", "valid"),
        ("+44 20 1234 5678", "valid"),
        ("123", "invalid"),
        ("abc123", "invalid"),
        ("invalid_phone", "invalid"),
        ("+", "invalid"),
    ]

    # Train
    validator.train(training_data, "phone")

    # Save
    validator.save("models/phone_validator.pkl")

    # Test
    test_cases = [
        "+1234567890",
        "123456",
        "invalid",
    ]

    print("\nValidation Results:")
    print("-" * 70)
    for text in test_cases:
        is_valid, confidence = validator.validate(text)
        status = "VALID" if is_valid else "INVALID"
        print(f"{text:<20} -> {status} ({confidence:.1%} confidence)")
