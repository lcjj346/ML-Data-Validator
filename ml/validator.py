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

    def explain_invalidity(self, text: str) -> str:
        """
        Explain why a text is considered invalid.

        Args:
            text: Text to analyze

        Returns:
            Human-readable explanation of what's wrong
        """
        if not self.is_trained:
            return "Model not trained"

        # Extract features
        features = self.feature_extractor.extract_features(text)

        # Get feature names for reference
        feature_names = [
            'length', 'word_count', 'comma_parts',
            'digit_ratio', 'letter_ratio', 'space_ratio', 'uppercase_ratio', 'lowercase_ratio',
            'count_plus', 'count_at', 'count_dot', 'count_hash', 'count_dash',
            'count_underscore', 'count_lparen', 'count_rparen', 'count_slash',
            'starts_uppercase', 'starts_digit', 'starts_plus', 'ends_digit', 'ends_letter',
            'email_like', 'has_single_at', 'has_username', 'domain_has_dot', 'domain_has_extension', 'common_domain',
            'phone_like', 'mixed_alphanum',
            'digit_sequences', 'capitalized_words', 'long_digits',
            'has_blk', 'has_ave', 'has_road', 'has_street', 'has_singapore',
            'has_com', 'has_net', 'has_org', 'has_edu',
            'has_at', 'has_plus', 'has_hash',
            'repeated_chars', 'triple_chars', 'max_bigram_freq', 'char_variety',
            'vowel_ratio', 'max_consecutive_consonants'
        ]

        # Analyze features to determine issues
        issues = []

        # Length checks
        if features[0] < 3:  # length
            issues.append("too short")
        elif features[0] > 100:
            issues.append("too long")

        # Data type specific checks based on training data type
        data_type_lower = self.data_type.lower()

        if 'email' in data_type_lower:
            if features[9] == 0:  # count_at
                issues.append("missing '@' symbol")
            elif features[9] > 1:
                issues.append("multiple '@' symbols")
            if features[10] == 0:  # count_dot
                issues.append("missing domain extension")
            if features[23] == 0:  # email_like pattern
                issues.append("invalid email format")

        elif 'phone' in data_type_lower:
            digit_ratio = features[3]
            if digit_ratio < 0.5:
                issues.append(f"insufficient digits ({digit_ratio*100:.0f}% digits)")
            if features[8] == 0 and features[0] > 8:  # No + sign for longer numbers
                issues.append("missing country code")
            if features[0] < 8:
                issues.append("too short for phone number")

        elif 'name' in data_type_lower:
            if features[1] < 2:  # word_count
                issues.append("missing first or last name")
            if features[3] > 0.1:  # digit_ratio > 10%
                issues.append("contains numbers")
            if features[6] > 0.8:  # uppercase_ratio
                issues.append("too many uppercase letters")
            if features[17] == 0:  # starts_uppercase
                issues.append("should start with capital letter")

        elif 'country' in data_type_lower or 'location' in data_type_lower:
            if features[1] > 5:  # word_count
                issues.append("too many words for country name")
            if features[3] > 0.2:  # digit_ratio
                issues.append("contains numbers")

        # General pattern issues
        if features[46] > 0:  # triple_chars (aaa, bbb)
            issues.append("repeated characters")

        # Only flag very unusual vowel/consonant patterns (more lenient)
        if features[4] > 0.5:  # Only check if text has significant letters
            if features[49] > 0.95:  # vowel_ratio extremely high
                issues.append("unusual vowel pattern")
            elif features[49] < 0.05:  # vowel_ratio extremely low
                issues.append("unusual consonant pattern")

        if features[50] > 6:  # max_consecutive_consonants (increased threshold)
            issues.append("too many consecutive consonants")

        # Special character issues
        special_chars = sum([features[i] for i in [11, 12, 13, 14, 15, 16]])  # hash, dash, underscore, parens, slash
        if special_chars > 10:
            issues.append("too many special characters")

        # If no specific issues found, provide generic message
        if not issues:
            # Check confidence level
            _, confidence = self.validate(text)
            if confidence < 0.6:
                issues.append("pattern doesn't match training data")
            else:
                issues.append("unusual pattern detected")

        # Format the explanation
        if len(issues) == 1:
            return issues[0].capitalize()
        elif len(issues) == 2:
            return f"{issues[0].capitalize()} and {issues[1]}"
        else:
            return f"{issues[0].capitalize()}, {', '.join(issues[1:-1])}, and {issues[-1]}"

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
