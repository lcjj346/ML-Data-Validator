"""
Generic ML Validator - Works for ANY data type

Train once on YOUR data, validate forever.
Works for: phone, email, address, name, or ANY custom column type.

Key: Uses YOUR training data, not pre-trained models.
"""

import os
import joblib
from typing import List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
        self.model = LogisticRegression(max_iter=5000, random_state=42, class_weight='balanced', solver='lbfgs')
        self.scaler = StandardScaler()
        self.feature_extractor = GenericFeatureExtractor()
        self.is_trained = False
        self.model_path = model_path
        self.data_type = "unknown"
        self.valid_whitelist = set()  # Store valid examples for exact matching
        self.training_metrics = {}  # Store train/test metrics

        # Try to load if path provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def train(self, training_data: List[Tuple[str, str]], data_type: str = "custom", test_size: float = 0.2):
        """
        Train validator on YOUR data with train/test split for proper evaluation.

        Args:
            training_data: List of (text, label) tuples
                          label should be "valid" or "invalid"
            data_type: Name of data type (e.g., "phone", "email", "address")
            test_size: Fraction of data to use for testing (default 0.2 = 20%)

        Returns:
            dict: Training metrics including accuracy, precision, recall, F1, confusion matrix

        Example:
            training_data = [
                ("+1234567890", "valid"),
                ("123", "invalid"),
                ("invalid_phone", "invalid"),
            ]
            metrics = validator.train(training_data, "phone")
        """
        print(f"Training Generic ML Validator for '{data_type}'...")
        print(f"Training examples: {len(training_data)}")

        # Build whitelist of valid examples (case-insensitive)
        self.valid_whitelist = set(
            str(text).lower().strip()
            for text, label in training_data
            if label.lower() == "valid"
        )

        # Extract features
        X = [self.feature_extractor.extract_features(text) for text, label in training_data]
        y = [1 if label.lower() == "valid" else 0 for text, label in training_data]

        # Check for class imbalance and warn
        valid_count = sum(y)
        invalid_count = len(y) - valid_count
        total = len(y)
        valid_pct = valid_count / total * 100
        invalid_pct = invalid_count / total * 100

        print(f"Class distribution: {valid_count} valid ({valid_pct:.1f}%), {invalid_count} invalid ({invalid_pct:.1f}%)")

        if valid_pct < 20 or valid_pct > 80:
            print(f"WARNING: Imbalanced training data detected! Consider adding more {'valid' if valid_pct < 20 else 'invalid'} examples.")

        if total < 50:
            print(f"WARNING: Small training set ({total} examples). Consider adding more examples for better accuracy.")

        # Initialize metrics dict
        self.training_metrics = {
            'total_samples': total,
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'used_split': False,
            'small_dataset_warning': False
        }

        min_class_count = min(valid_count, invalid_count)

        # Decide whether to use train/test split
        # Need at least 10 samples and 4 per class for stratified split
        if total >= 10 and min_class_count >= 4:
            # Perform train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )

            self.training_metrics['used_split'] = True
            self.training_metrics['train_size'] = len(X_train)
            self.training_metrics['test_size'] = len(X_test)

            print(f"Train/test split: {len(X_train)} train, {len(X_test)} test")

            # Fit scaler on train data only, transform both
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model on train set only
            self.model.fit(X_train_scaled, y_train)

            # Evaluate on train set
            y_train_pred = self.model.predict(X_train_scaled)
            train_metrics = self._calculate_metrics(y_train, y_train_pred, prefix='train')

            # Evaluate on test set
            y_test_pred = self.model.predict(X_test_scaled)
            test_metrics = self._calculate_metrics(y_test, y_test_pred, prefix='test')

            # Store metrics
            self.training_metrics.update(train_metrics)
            self.training_metrics.update(test_metrics)

            print(f"Train accuracy: {train_metrics['train_accuracy']:.2%}")
            print(f"Test accuracy: {test_metrics['test_accuracy']:.2%}")
            print(f"Test F1 score: {test_metrics['test_f1']:.2%}")

        else:
            # Dataset too small for split - train on all data
            self.training_metrics['small_dataset_warning'] = True
            print(f"WARNING: Dataset too small for train/test split (need >= 10 samples, >= 4 per class)")
            print("Training on all data - metrics may be overfit")

            # Scale all features
            X_scaled = self.scaler.fit_transform(X)

            # Train model on all data
            self.model.fit(X_scaled, y)

            # Calculate training metrics (on same data - will be overfit)
            y_pred = self.model.predict(X_scaled)
            train_metrics = self._calculate_metrics(y, y_pred, prefix='train')
            self.training_metrics.update(train_metrics)

            print(f"Training accuracy: {train_metrics['train_accuracy']:.2%}")

        self.is_trained = True
        self.data_type = data_type
        print("Training complete!")

        return self.training_metrics

    def _calculate_metrics(self, y_true, y_pred, prefix=''):
        """Calculate accuracy, precision, recall, F1, and confusion matrix."""
        metrics = {}
        key_prefix = f"{prefix}_" if prefix else ""

        metrics[f'{key_prefix}accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{key_prefix}precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f'{key_prefix}recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics[f'{key_prefix}f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics[f'{key_prefix}confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        return metrics

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

        # First check whitelist (exact match, case-insensitive)
        text_lower = str(text).lower().strip()
        if self.valid_whitelist and text_lower in self.valid_whitelist:
            return True, 1.0  # Exact match = definitely valid

        # Otherwise use ML model
        features = self.feature_extractor.extract_features(text)
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
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

        results = []
        texts_for_ml = []
        ml_indices = []

        # First pass: check whitelist
        for i, text in enumerate(texts):
            text_lower = str(text).lower().strip()
            if self.valid_whitelist and text_lower in self.valid_whitelist:
                results.append((True, 1.0))  # Exact match
            else:
                results.append(None)  # Placeholder for ML
                texts_for_ml.append(text)
                ml_indices.append(i)

        # Second pass: ML for non-whitelist entries
        if texts_for_ml:
            X = [self.feature_extractor.extract_features(text) for text in texts_for_ml]
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)

            for idx, pred, prob in zip(ml_indices, predictions, probabilities):
                is_valid = bool(pred)
                confidence = float(max(prob))
                results[idx] = (is_valid, confidence)

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

        # Helper to detect data type category with multiple keywords
        def is_type(keywords):
            return any(kw in data_type_lower for kw in keywords)

        if is_type(['email', 'mail', 'e-mail', 'e_mail']):
            if features[9] == 0:  # count_at
                issues.append("missing '@' symbol")
            elif features[9] > 1:
                issues.append("multiple '@' symbols")
            if features[10] == 0:  # count_dot
                issues.append("missing domain extension")
            if features[23] == 0:  # email_like pattern
                issues.append("invalid email format")

        elif is_type(['phone', 'mobile', 'cell', 'tel', 'contact_number', 'phone_number']):
            digit_ratio = features[3]
            if digit_ratio < 0.5:
                issues.append(f"insufficient digits ({digit_ratio*100:.0f}% digits)")
            if features[8] == 0 and features[0] > 8:  # No + sign for longer numbers
                issues.append("missing country code")
            if features[0] < 8:
                issues.append("too short for phone number")

        elif is_type(['name', 'person', 'full_name', 'fullname', 'customer_name']):
            if features[1] < 2:  # word_count
                issues.append("missing first or last name")
            if features[3] > 0.1:  # digit_ratio > 10%
                issues.append("contains numbers")
            if features[6] > 0.8:  # uppercase_ratio
                issues.append("too many uppercase letters")
            if features[17] == 0:  # starts_uppercase
                issues.append("should start with capital letter")

        elif is_type(['country', 'location', 'nation', 'region', 'territory']):
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
            'scaler': self.scaler,
            'data_type': self.data_type,
            'is_trained': self.is_trained,
            'valid_whitelist': self.valid_whitelist,
            'training_metrics': self.training_metrics,
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
            self.scaler = model_data.get('scaler', StandardScaler())  # Backward compatibility
            self.data_type = model_data['data_type']
            self.is_trained = model_data['is_trained']
            self.valid_whitelist = model_data.get('valid_whitelist', set())  # Load whitelist
            self.training_metrics = model_data.get('training_metrics', {})  # Load metrics
            self.model_path = filepath
            print(f"Model loaded: {self.data_type} validator")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


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
