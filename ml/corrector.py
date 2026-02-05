"""
Generic ML Corrector - Works for ANY data type

Learns correction patterns from YOUR training data.
Works for: phone, email, address, name, or ANY custom column type.

Uses similarity-based matching to suggest corrections.
"""

import os
import joblib
from typing import List, Optional
import difflib


class GenericMLCorrector:
    """
    One corrector for ALL data types.

    Learns from examples:
    - "1234567890" → "+11234567890" (phone)
    - "user@gmial.com" → "user@gmail.com" (email typo)
    - "Sinapore" → "Singapore" (location typo)
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize generic corrector.

        Args:
            model_path: Path to load trained model (optional)
        """
        self.is_trained = False
        self.model_path = model_path
        self.data_type = "unknown"
        self.valid_examples = []  # Store valid examples for similarity matching
        self.similarity_threshold = 0.6  # Minimum similarity to suggest (0.0-1.0)

        # Try to load if path provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def correct(self, text: str) -> Optional[str]:
        """
        Correct invalid text using similarity matching.

        Args:
            text: Invalid text to correct

        Returns:
            Corrected text (closest valid example), or None if no good match
        """
        if not self.is_trained or not self.valid_examples:
            return None

        # Don't suggest corrections for empty, NaN, or whitespace-only strings
        if not text or str(text).strip() == '' or str(text).lower() in ['nan', 'none', 'null']:
            return None

        # Convert text to string for comparison
        text_str = str(text)
        text_lower = text_str.lower()

        # Find all matches above threshold
        candidates = []
        for valid_example in self.valid_examples:
            valid_example_str = str(valid_example)
            # Calculate similarity using difflib's SequenceMatcher
            similarity = difflib.SequenceMatcher(None, text_lower, valid_example_str.lower()).ratio()

            if similarity >= self.similarity_threshold:
                candidates.append((valid_example_str, similarity))

        if not candidates:
            return None

        # Sort candidates by: 1) similarity (desc), 2) canonical form preference
        def canonical_score(s):
            """Prefer canonical forms based on length and casing"""
            if len(s) <= 3 and s.isupper():
                return 0  # Best for short: "USA", "UK", "UAE"
            elif s.istitle():
                return 1  # Best for long: "Singapore", "Malaysia"
            elif s[0].isupper() if s else False:
                return 2  # OK: "United States"
            elif s.isupper():
                return 3  # Less preferred for long: "SINGAPORE"
            else:
                return 4  # Least preferred: "usa", "UsA"

        candidates.sort(key=lambda x: (-x[1], canonical_score(x[0]), len(x[0])))
        best_match = candidates[0][0]

        # Only return if not identical to input
        if text_str != best_match:
            return best_match

        return None

    def train(self, valid_examples: List[str], data_type: str = "custom"):
        """
        Train corrector with valid examples.

        Args:
            valid_examples: List of valid text examples
            data_type: Name of data type
        """
        print(f"Training corrector for '{data_type}' with {len(valid_examples)} valid examples...")

        # Store unique valid examples
        self.valid_examples = list(set(valid_examples))
        self.data_type = data_type
        self.is_trained = True

        print(f"Corrector trained with {len(self.valid_examples)} unique examples")

    def save(self, filepath: str):
        """Save corrector model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'data_type': self.data_type,
            'is_trained': self.is_trained,
            'valid_examples': self.valid_examples,
            'similarity_threshold': self.similarity_threshold,
        }

        joblib.dump(model_data, filepath)
        self.model_path = filepath
        print(f"Corrector saved to {filepath}")

    def load(self, filepath: str) -> bool:
        """Load corrector model"""
        if not os.path.exists(filepath):
            print(f"Corrector file not found: {filepath}")
            return False

        try:
            model_data = joblib.load(filepath)

            self.data_type = model_data['data_type']
            self.is_trained = model_data['is_trained']
            self.valid_examples = model_data.get('valid_examples', [])
            self.similarity_threshold = model_data.get('similarity_threshold', 0.6)
            self.model_path = filepath

            print(f"Corrector loaded: {self.data_type} with {len(self.valid_examples)} examples")
            return True
        except Exception as e:
            print(f"Error loading corrector: {e}")
            return False


if __name__ == "__main__":
    # Example: Train a corrector
    print("=" * 70)
    print("GENERIC ML CORRECTOR - TEST")
    print("=" * 70)

    corrector = GenericMLCorrector()

    # Test simple typo correction
    test_cases = [
        "1234567e90",  # e → 3
        "PhOne",  # O → 0
        "emaiI",  # I → 1
    ]

    print("\nCorrection Results:")
    print("-" * 70)
    for text in test_cases:
        corrected = corrector.correct(text)
        if corrected:
            print(f"{text:<20} -> {corrected}")
        else:
            print(f"{text:<20} -> (no correction)")
