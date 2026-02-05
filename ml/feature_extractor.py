"""
Generic Feature Extractor - Works for ANY text data type

Extracts features from text that can be used to train validators for:
- Phone numbers
- Email addresses
- Addresses
- Names
- Any custom text data

The key: Extract GENERIC patterns, not hardcoded rules.
"""

from typing import List
import re
import math


class GenericFeatureExtractor:
    """Extract features from any text for ML training"""

    @staticmethod
    def extract_features(text: str, column_name: str = None) -> List[float]:
        """
        Extract generic features from text.

        Args:
            text: Any text string
            column_name: Optional column name for future column-aware features

        Returns:
            List of numeric features for ML
        """
        # column_name reserved for future column-aware enhancements
        if not text or not isinstance(text, str):
            text = str(text) if text is not None else ""

        text = text.strip()
        length = len(text)

        features = []

        # ========== LENGTH FEATURES ==========
        features.append(length)
        features.append(len(text.split()))  # Word count
        features.append(len(text.split(',')))  # Comma-separated parts

        # ========== CHARACTER TYPE RATIOS ==========
        if length > 0:
            features.append(sum(c.isdigit() for c in text) / length)  # Digit ratio
            features.append(sum(c.isalpha() for c in text) / length)  # Letter ratio
            features.append(sum(c.isspace() for c in text) / length)  # Space ratio
            features.append(sum(c.isupper() for c in text) / length)  # Uppercase ratio
            features.append(sum(c.islower() for c in text) / length)  # Lowercase ratio
        else:
            features.extend([0, 0, 0, 0, 0])

        # ========== SPECIAL CHARACTER COUNTS ==========
        features.append(text.count('+'))
        features.append(text.count('@'))
        features.append(text.count('.'))
        features.append(text.count('#'))
        features.append(text.count('-'))
        features.append(text.count('_'))
        features.append(text.count('('))
        features.append(text.count(')'))
        features.append(text.count('/'))

        # ========== POSITION FEATURES ==========
        features.append(1 if text and text[0].isupper() else 0)  # Starts with uppercase
        features.append(1 if text and text[0].isdigit() else 0)  # Starts with digit
        features.append(1 if text and text[0] == '+' else 0)  # Starts with +
        features.append(1 if text and text[-1].isdigit() else 0)  # Ends with digit
        features.append(1 if text and text[-1].isalpha() else 0)  # Ends with letter

        # ========== PATTERN FEATURES ==========
        # Email patterns
        has_at = '@' in text
        has_dot = '.' in text
        features.append(1 if has_at and has_dot else 0)  # Email-like

        # More detailed email features (always add 5 features)
        if has_at:
            parts = text.split('@')
            features.append(1 if len(parts) == 2 else 0)  # Exactly one @
            if len(parts) == 2:
                username, domain = parts
                features.append(1 if len(username) > 0 else 0)  # Has username
                features.append(1 if '.' in domain else 0)  # Domain has dot
                features.append(1 if domain.count('.') >= 1 else 0)  # Domain has extension
                # Common email domains
                common_domains = ['gmail', 'yahoo', 'outlook', 'hotmail', 'company', 'example']
                features.append(1 if any(d in domain.lower() for d in common_domains) else 0)
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0])

        # Phone patterns
        features.append(1 if text.startswith('+') and text[1:].replace(' ', '').replace('-', '').isdigit() else 0)  # Phone-like

        # Mixed patterns
        features.append(1 if any(char.isdigit() for char in text) and any(char.isalpha() for char in text) else 0)  # Mixed alphanumeric

        # ========== REGEX PATTERNS ==========
        features.append(len(re.findall(r'\d+', text)))  # Number of digit sequences
        features.append(len(re.findall(r'[A-Z][a-z]+', text)))  # Number of capitalized words
        features.append(len(re.findall(r'\b\d{4,}\b', text)))  # Long digit sequences (4+)

        # ========== COMMON KEYWORDS (learned from data) ==========
        # These will be learned patterns from your training data
        common_patterns = [
            'blk', 'ave', 'road', 'street', 'singapore',  # Address keywords
            'com', 'net', 'org', 'edu',  # Domain keywords
            '@', '+', '#',  # Structural markers
        ]

        for pattern in common_patterns:
            features.append(1 if pattern.lower() in text.lower() else 0)

        # ========== CHARACTER N-GRAMS (for typo detection) ==========
        # Character bigrams and trigrams help detect valid vs invalid patterns
        text_lower = text.lower()

        # Count repeated characters (aa, bb, cc, etc.)
        repeated_chars = sum(1 for i in range(len(text_lower) - 1) if text_lower[i] == text_lower[i + 1])
        features.append(repeated_chars)

        # Count triple repeated characters (aaa, bbb, etc.)
        triple_chars = sum(1 for i in range(len(text_lower) - 2)
                          if text_lower[i] == text_lower[i + 1] == text_lower[i + 2])
        features.append(triple_chars)

        # Unusual character sequences (repeated bigrams like "ee", "aa" multiple times)
        bigram_counts = {}
        for i in range(len(text_lower) - 1):
            bigram = text_lower[i:i+2]
            if bigram.isalpha():
                bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        max_bigram_freq = max(bigram_counts.values()) if bigram_counts else 0
        features.append(max_bigram_freq)  # How many times the most common bigram appears

        # Character variety ratio (unique chars / total chars)
        if length > 0:
            unique_chars = len(set(text_lower))
            char_variety = unique_chars / length
            features.append(char_variety)
        else:
            features.append(0)

        # Consonant/vowel patterns (abnormal patterns indicate typos)
        vowels = set('aeiou')
        if length > 0:
            vowel_ratio = sum(1 for c in text_lower if c in vowels) / length
            features.append(vowel_ratio)

            # Count consecutive consonants (more than 3 is unusual in English)
            max_consecutive_consonants = 0
            current_consonants = 0
            for c in text_lower:
                if c.isalpha() and c not in vowels:
                    current_consonants += 1
                    max_consecutive_consonants = max(max_consecutive_consonants, current_consonants)
                else:
                    current_consonants = 0
            features.append(max_consecutive_consonants)
        else:
            features.append(0)
            features.append(0)

        # ========== NUMERIC VALUE FEATURES ==========
        # Enhanced numeric features for better range validation (blood sugar, age, weight, etc.)
        # These features are GENERIC and work for any numeric column
        try:
            numeric_value = float(text)
            # Check if it's NaN (special float value)
            if math.isnan(numeric_value) or math.isinf(numeric_value):
                # Not a valid number - add dummy features
                features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                # Feature 1: The actual numeric value (most important!)
                features.append(numeric_value)
                # Feature 2: Is numeric flag
                features.append(1)
                # Feature 3: Squared value (helps with non-linear boundaries)
                features.append(numeric_value ** 2)
                # Feature 4: Cubed value (captures higher order patterns)
                features.append(numeric_value ** 3)
                # Feature 5: Square root (if positive)
                features.append(math.sqrt(abs(numeric_value)) if numeric_value >= 0 else 0)
                # Feature 6: Log value (helps with exponential patterns)
                features.append(math.log(abs(numeric_value) + 1))
                # Feature 7: Inverse (1/x) - helps with reciprocal patterns
                features.append(1 / numeric_value if numeric_value != 0 else 0)

                # Feature 8: Sign (positive/negative/zero)
                features.append(1 if numeric_value > 0 else (-1 if numeric_value < 0 else 0))
                # Feature 9: Absolute value
                features.append(abs(numeric_value))

                # Features 10-19: Generic decimal bucket features (works for any range)
                # These create decision boundaries at different magnitudes
                features.append(1 if numeric_value < 0 else 0)  # Negative
                features.append(1 if 0 <= numeric_value < 1 else 0)  # [0, 1)
                features.append(1 if 1 <= numeric_value < 2 else 0)  # [1, 2)
                features.append(1 if 2 <= numeric_value < 5 else 0)  # [2, 5)
                features.append(1 if 5 <= numeric_value < 10 else 0)  # [5, 10)
                features.append(1 if 10 <= numeric_value < 20 else 0)  # [10, 20)
                features.append(1 if 20 <= numeric_value < 50 else 0)  # [20, 50)
                features.append(1 if 50 <= numeric_value < 100 else 0)  # [50, 100)
                features.append(1 if 100 <= numeric_value < 200 else 0)  # [100, 200)
                features.append(1 if numeric_value >= 200 else 0)  # [200, +inf)

                # Feature 20: Number of decimal places (precision indicator)
                text_str = str(text).strip()
                if '.' in text_str:
                    decimal_places = len(text_str.split('.')[1])
                    features.append(min(decimal_places, 5))  # Cap at 5
                else:
                    features.append(0)

        except (ValueError, TypeError):
            # Not a number - add dummy features (20 features)
            features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # ========== TOTAL: ~67 FEATURES ==========

        return features


if __name__ == "__main__":
    # Test the feature extractor
    extractor = GenericFeatureExtractor()

    test_cases = [
        "+1234567890",  # Phone
        "user@example.com",  # Email
        "Blk 123 Ang Mo Kio Ave 3 #01-234",  # Singapore address
        "John Doe",  # Name
    ]

    print("Testing Generic Feature Extractor:")
    print("=" * 60)

    for text in test_cases:
        features = extractor.extract_features(text)
        print(f"\nText: {text}")
        print(f"Features: {len(features)} features extracted")
        print(f"First 10 features: {features[:10]}")

    print("\n" + "=" * 60)
    print(f"Total features: {len(extractor.extract_features('test'))}")
