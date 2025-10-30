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


class GenericFeatureExtractor:
    """Extract features from any text for ML training"""

    @staticmethod
    def extract_features(text: str) -> List[float]:
        """
        Extract generic features from text.

        Args:
            text: Any text string

        Returns:
            List of numeric features for ML
        """
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

        # ========== TOTAL: ~47 FEATURES ==========

        return features

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get feature names for debugging/analysis"""
        return [
            'length', 'word_count', 'comma_parts',
            'digit_ratio', 'letter_ratio', 'space_ratio', 'uppercase_ratio', 'lowercase_ratio',
            'count_plus', 'count_at', 'count_dot', 'count_hash', 'count_dash',
            'count_underscore', 'count_lparen', 'count_rparen', 'count_slash',
            'starts_uppercase', 'starts_digit', 'starts_plus', 'ends_digit', 'ends_letter',
            'email_like', 'phone_like', 'mixed_alphanum',
            'digit_sequences', 'capitalized_words', 'long_digits',
            'has_blk', 'has_ave', 'has_road', 'has_street', 'has_singapore',
            'has_com', 'has_net', 'has_org', 'has_edu',
            'has_at', 'has_plus', 'has_hash',
        ]


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
