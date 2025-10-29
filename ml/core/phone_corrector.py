"""
Edit Distance Phone Corrector - ML-based character-level phone number correction.

Uses XGBoost to learn character-level edit operations for correcting invalid phone numbers:
- Keep character as-is
- Delete character
- Replace with specific digit (0-9)
- Insert country code or + sign

Example usage:
    from ml.edit_distance_corrector import EditDistanceCorrector

    corrector = EditDistanceCorrector()
    corrector.train_from_csv('data/xgboost_corrector_training.csv')
    corrector.save('saved_models/edit_distance_corrector.pkl')

    corrected = corrector.correct_phone("1234567e90")
    print(corrected)  # Output: +11234567890
"""

import os
import pickle
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import xgboost as xgb


class EditDistanceCorrector:
    """
    XGBoost-based phone number corrector using character-level edit operations.

    This class learns to apply edit operations (keep, delete, replace, insert) to
    correct invalid phone numbers at the character level.

    Attributes:
        model: Trained XGBoost classifier
        label_encoder: Maps operation names to integer labels
        label_decoder: Maps integer labels back to operation names
        typo_map: Common letter-to-digit typo mappings
        inference_label_map: Mapping for remapped labels during inference
    """

    def __init__(self):
        """Initialize EditDistanceCorrector with operation mappings."""
        self.model: Optional[xgb.XGBClassifier] = None
        self.label_encoder: Dict[str, int] = {
            'keep': 0,
            'delete': 1,
            'replace_0': 2, 'replace_1': 3, 'replace_2': 4, 'replace_3': 5,
            'replace_4': 6, 'replace_5': 7, 'replace_6': 8, 'replace_7': 9,
            'replace_8': 10, 'replace_9': 11,
            'insert_+': 12,
            'insert_country': 13,
        }
        self.label_decoder: Dict[int, str] = {v: k for k, v in self.label_encoder.items()}

        # Letter to digit mapping for common typos
        self.typo_map: Dict[str, str] = {
            'o': '0', 'O': '0',
            'l': '1', 'L': '1', 'I': '1', 'i': '1',
            'e': '3', 'E': '3',
            's': '5', 'S': '5',
            'b': '8', 'B': '8',
            'g': '9', 'G': '9',
        }
        self.inference_label_map: Dict[int, int] = {}

    def extract_features(self, phone_str: str, position: int) -> List[int]:
        """
        Extract features for a character at a given position.

        Extracts 16 features describing the character and its context:
        - Character type (digit, alpha, +, space, special)
        - Position information (index, total length, first/last)
        - Context (left/right characters)
        - Global features (starts with +, digit count)
        - Typo likelihood

        Args:
            phone_str: Phone number string
            position: Character position (0-indexed)

        Returns:
            List of 16 integer features
        """
        features = []

        # Character at position
        char = phone_str[position] if position < len(phone_str) else ''

        # Character type features (5 features)
        features.append(1 if char.isdigit() else 0)
        features.append(1 if char.isalpha() else 0)
        features.append(1 if char == '+' else 0)
        features.append(1 if char == ' ' else 0)
        features.append(1 if char in '-()./' else 0)

        # Position features (4 features)
        features.append(position)
        features.append(len(phone_str))
        features.append(1 if position == 0 else 0)
        features.append(1 if position == len(phone_str) - 1 else 0)

        # Context features - left (2 features)
        left_char = phone_str[position - 1] if position > 0 else ''
        features.append(1 if left_char.isdigit() else 0)
        features.append(1 if left_char == '+' else 0)

        # Context features - right (2 features)
        right_char = phone_str[position + 1] if position < len(phone_str) - 1 else ''
        features.append(1 if right_char.isdigit() else 0)
        features.append(1 if right_char.isalpha() else 0)

        # Global features (2 features)
        features.append(1 if phone_str.startswith('+') else 0)
        digit_count = sum(1 for c in phone_str[:position] if c.isdigit())
        features.append(digit_count)

        # Typo likelihood (1 feature)
        features.append(1 if char in self.typo_map else 0)

        return features

    def prepare_training_data(self, training_data):
        """Convert to features and labels"""
        X = []
        y = []

        for example in training_data:
            invalid = example['invalid']
            operations = example['operations']

            for pos in range(len(invalid)):
                features = self.extract_features(invalid, pos)
                label = operations[pos] if pos < len(operations) else 'keep'

                X.append(features)
                # Map label to encoded value, default to 'keep' if not found
                if label in self.label_encoder:
                    y.append(self.label_encoder[label])
                elif label.startswith('replace_') and len(label) > 8:
                    # Handle any replace operation
                    digit = label.split('_')[1]
                    if f'replace_{digit}' in self.label_encoder:
                        y.append(self.label_encoder[f'replace_{digit}'])
                    else:
                        y.append(self.label_encoder['keep'])
                else:
                    y.append(self.label_encoder['keep'])

        return np.array(X), np.array(y)

    def train_from_csv(self, csv_path='../data/xgboost_corrector_training.csv'):
        """Train the edit distance model from saved CSV data"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Training data not found at {csv_path}\n"
                f"Please run 'python generate_training_data.py' first to create the training data."
            )

        print(f"Loading training data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Convert DataFrame to training_data format
        training_data = []
        for _, row in df.iterrows():
            training_data.append({
                'invalid': row['invalid'],
                'valid': row['valid'],
                'operations': row['operations'].split('|')  # Convert pipe-separated string back to list
            })

        print(f"Loaded {len(training_data)} training examples")

        print(f"Sample examples:")
        for i in range(min(3, len(training_data))):
            print(f"  '{training_data[i]['invalid']}' -> '{training_data[i]['valid']}'")

        print("\nPreparing features...")
        X, y = self.prepare_training_data(training_data)

        print(f"Feature matrix: {X.shape}")
        print(f"Labels: {y.shape}")

        print("\nTraining XGBoost model...")
        # Remap labels to be consecutive (XGBoost requirement)
        unique_labels = np.unique(y)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y_remapped = np.array([label_map[label] for label in y])

        # Store reverse mapping for inference
        self.inference_label_map = {new: old for old, new in label_map.items()}

        print(f"Number of unique classes: {len(unique_labels)}")

        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=len(unique_labels),
            random_state=42
        )

        self.model.fit(X, y_remapped)

        # Evaluate
        accuracy = self.model.score(X, y)
        print(f"\nTraining accuracy: {accuracy:.3f}")

        print("Training complete!")

    def correct_phone(self, invalid_phone):
        """Correct an invalid phone number with strict validation"""
        if not self.model:
            return None

        # Extract features for each position
        predictions = []
        for pos in range(len(invalid_phone)):
            features = self.extract_features(invalid_phone, pos)
            pred_remapped = self.model.predict([features])[0]
            # Map back to original label
            if hasattr(self, 'inference_label_map'):
                pred = self.inference_label_map[pred_remapped]
            else:
                pred = pred_remapped
            operation = self.label_decoder[pred]
            predictions.append(operation)

        # Apply edit operations - collect digits and special chars
        result = []

        # Check structure first
        digit_count = sum(1 for c in invalid_phone if c.isdigit())
        has_plus = invalid_phone.strip().startswith('+')
        has_letters = any(c.isalpha() for c in invalid_phone)

        # If too many letters, return None
        letter_count = sum(1 for c in invalid_phone if c.isalpha())
        if letter_count > 3:  # Too corrupted
            return None

        # Process each character
        for pos, (char, op) in enumerate(zip(invalid_phone, predictions)):
            if op == 'keep':
                # Only keep if it's a digit, +, or we're fixing typos
                if char.isdigit():
                    result.append(char)
                elif char == '+' and not result:  # + only at start
                    result.append(char)
                elif char in self.typo_map:
                    # Replace typo with correct digit
                    result.append(self.typo_map[char])
                # Skip everything else (spaces, parentheses, dashes, letters)
            elif op == 'delete':
                # Skip character
                continue
            elif op.startswith('replace_'):
                # Replace with digit
                digit = op.split('_')[1]
                result.append(digit)
            elif op == 'insert_country':
                # Keep the digit
                if char.isdigit():
                    result.append(char)

        corrected = ''.join(result)

        # Add country code if needed
        if not corrected.startswith('+'):
            # Count digits in corrected string
            final_digit_count = sum(1 for c in corrected if c.isdigit())

            # Special case: if 11 digits starting with '1', it's US (1 + 10 digits)
            if final_digit_count == 11 and corrected.startswith('1'):
                corrected = '+' + corrected
            # If 10 digits starting with 1, could be US with country code already
            elif final_digit_count == 10 and corrected.startswith('1'):
                # Assume the 1 is already the country code
                corrected = '+' + corrected
            # If 10 digits not starting with 1, assume US without country code
            elif final_digit_count == 10:
                corrected = '+1' + corrected
            # If 11 digits (not starting with 1), assume China
            elif final_digit_count == 11:
                corrected = '+86' + corrected
            # If 9 digits, assume Australia
            elif final_digit_count == 9:
                corrected = '+61' + corrected
            else:
                # Don't know what country, return None
                return None

        # Strict validation: must be only + and digits
        if not corrected or not corrected.startswith('+'):
            return None

        # Check if result contains only + and digits
        if not all(c.isdigit() or c == '+' for c in corrected):
            return None

        # Check length (must be between 10-15 characters including +)
        if len(corrected) < 10 or len(corrected) > 16:
            return None

        # Check that + only appears at start
        if corrected.count('+') != 1:
            return None

        return corrected

    def save(self, filepath='saved_models/edit_distance_corrector.pkl'):
        """Save model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'label_decoder': self.label_decoder,
                'typo_map': self.typo_map,
                'inference_label_map': getattr(self, 'inference_label_map', {})
            }, f)
        print(f"\nModel saved to {filepath}")

    def load(self, filepath='saved_models/edit_distance_corrector.pkl'):
        """Load model"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.label_encoder = data['label_encoder']
                self.label_decoder = data['label_decoder']
                self.typo_map = data['typo_map']
                self.inference_label_map = data.get('inference_label_map', {})
            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"Model not found at {filepath}")
            return False

    def is_loaded(self):
        return self.model is not None


if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING EDIT DISTANCE PHONE CORRECTOR")
    print("=" * 70)

    corrector = EditDistanceCorrector()
    corrector.train_from_csv()
    corrector.save()

    print("\n" + "=" * 70)
    print("TESTING MODEL")
    print("=" * 70)

    test_cases = [
        "1234567890",
        "+1234567e9o",
        "+44 123 456 789",
        "441234567890",
        "+33 6ol 23 45 67",
    ]

    print("\nResults:")
    for test in test_cases:
        result = corrector.correct_phone(test)
        print(f"  '{test}' -> '{result}'")
