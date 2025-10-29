"""
Universal NLP-Based Validator

This validator uses NLP techniques to validate ANY text column type without
requiring pre-trained models or pre-defined schemas. It works by:

1. Entity Recognition (spaCy) - Detects locations, persons, organizations
2. Similarity Analysis (Sentence Transformers) - Validates by comparing to column patterns
3. Fuzzy Matching (FuzzyWuzzy + pycountry) - Validates known entities like countries

Example usage:
    validator = UniversalNLPValidator()
    result = validator.validate("Sinapore", "Country", ["USA", "UK", "Canada"])
    # Returns: ValidationResult(is_valid=False, confidence=0.85, suggested="Singapore")
"""

import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import pycountry
import numpy as np
from typing import List, Optional, Tuple
import warnings

from ml.base_validator import BaseValidator, ValidationResult


class UniversalNLPValidator(BaseValidator):
    """
    Universal validator that uses NLP to validate any text column.

    This validator doesn't need pre-training. It infers data type from
    column name and sample values, then validates accordingly.

    Techniques used:
    - spaCy NER for entity detection
    - Sentence transformers for semantic similarity
    - Fuzzy matching for known reference databases
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the universal NLP validator."""
        super().__init__(model_path)

        try:
            # Load spaCy model (small, fast)
            self.spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            warnings.warn(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            self.spacy_nlp = None

        # Load sentence transformer (for semantic similarity)
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            warnings.warn(f"Could not load sentence transformer: {e}")
            self.sentence_model = None

        # Load reference databases
        self.countries = [c.name for c in pycountry.countries]
        self.country_codes = {c.alpha_2: c.name for c in pycountry.countries}

        # Add common variations
        self.countries.extend([
            "USA", "UK", "UAE", "US", "United States"
        ])

        self.is_trained = True  # No training needed

    def validate(self, value: str, column_name: str = "",
                 column_samples: List[str] = None) -> ValidationResult:
        """
        Validate a single value using NLP techniques.

        Args:
            value: The value to validate
            column_name: Name of the column (provides context)
            column_samples: Sample values from the same column (for pattern learning)

        Returns:
            ValidationResult with validation status and confidence
        """
        if column_samples is None:
            column_samples = []

        # Convert to string
        value_str = str(value).strip()

        # Empty value check
        if not value_str or value_str.lower() in ['nan', 'none', 'null', '']:
            return ValidationResult(
                is_valid=False,
                confidence=1.0,
                error_type="empty_value"
            )

        # Detect entity type from column name
        entity_type = self._detect_entity_type(column_name, column_samples)

        # Apply appropriate validation technique
        if entity_type == "location":
            return self._validate_location(value_str)
        elif entity_type == "person":
            return self._validate_person_name(value_str)
        elif entity_type == "organization":
            return self._validate_organization(value_str)
        elif entity_type == "country":
            return self._validate_country(value_str)
        else:
            # Unknown type - use similarity-based validation
            if column_samples:
                return self._validate_by_similarity(value_str, column_samples)
            else:
                # No context - assume valid
                return ValidationResult(
                    is_valid=True,
                    confidence=0.5,
                    error_type=None
                )

    def _detect_entity_type(self, column_name: str,
                           column_samples: List[str]) -> str:
        """
        Detect what type of entity this column contains.

        Args:
            column_name: Name of the column
            column_samples: Sample values from the column

        Returns:
            Entity type string (location, person, organization, country, unknown)
        """
        column_lower = column_name.lower()

        # Check column name patterns
        if any(word in column_lower for word in ['country', 'nation', 'nationality']):
            return "country"

        if any(word in column_lower for word in ['city', 'location', 'place', 'state', 'region']):
            return "location"

        if any(word in column_lower for word in ['name', 'person', 'customer', 'employee', 'user']):
            return "person"

        if any(word in column_lower for word in ['company', 'organization', 'org', 'business', 'vendor']):
            return "organization"

        # Analyze sample values with spaCy
        if self.spacy_nlp and column_samples:
            entity_types = []
            for sample in column_samples[:5]:  # Check first 5 samples
                doc = self.spacy_nlp(str(sample))
                for ent in doc.ents:
                    entity_types.append(ent.label_)

            if entity_types:
                # Most common entity type
                from collections import Counter
                most_common = Counter(entity_types).most_common(1)[0][0]

                if most_common == "GPE":  # Geopolitical entity
                    return "location"
                elif most_common == "PERSON":
                    return "person"
                elif most_common == "ORG":
                    return "organization"

        return "unknown"

    def _validate_country(self, value: str) -> ValidationResult:
        """
        Validate country names using fuzzy matching.

        Args:
            value: Country name to validate

        Returns:
            ValidationResult
        """
        # Check exact match first
        if value in self.countries:
            return ValidationResult(
                is_valid=True,
                confidence=1.0,
                error_type=None
            )

        # Check country codes
        if value.upper() in self.country_codes:
            return ValidationResult(
                is_valid=True,
                confidence=1.0,
                error_type=None
            )

        # Fuzzy match against all countries
        best_match = None
        best_score = 0

        for country in self.countries:
            score = fuzz.ratio(value.lower(), country.lower())
            if score > best_score:
                best_score = score
                best_match = country

        # Confidence threshold
        confidence = best_score / 100.0
        is_valid = confidence >= 0.95  # Very high threshold for exact matches

        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            error_type="fuzzy_match" if not is_valid else None,
            metadata={"best_match": best_match, "similarity": confidence}
        )

    def _validate_location(self, value: str) -> ValidationResult:
        """
        Validate location names using spaCy NER and fuzzy matching.

        Args:
            value: Location name to validate

        Returns:
            ValidationResult
        """
        if not self.spacy_nlp:
            return ValidationResult(is_valid=True, confidence=0.5)

        # First check if it's a country
        country_result = self._validate_country(value)
        if country_result.is_valid:
            return country_result

        # Use spaCy to detect if it looks like a location
        doc = self.spacy_nlp(value)

        # Check for GPE (geopolitical entity) or LOC (location)
        has_location_entity = any(
            ent.label_ in ["GPE", "LOC", "FAC"]
            for ent in doc.ents
        )

        if has_location_entity:
            return ValidationResult(
                is_valid=True,
                confidence=0.8,
                error_type=None
            )

        # Check if it looks like a place name (capitalized words)
        words = value.split()
        is_capitalized = all(word[0].isupper() for word in words if word)

        if is_capitalized and len(words) <= 4:
            return ValidationResult(
                is_valid=True,
                confidence=0.6,
                error_type=None
            )

        # Probably not a valid location
        return ValidationResult(
            is_valid=False,
            confidence=0.4,
            error_type="not_recognized_as_location"
        )

    def _validate_person_name(self, value: str) -> ValidationResult:
        """
        Validate person names using spaCy NER.

        Args:
            value: Person name to validate

        Returns:
            ValidationResult
        """
        if not self.spacy_nlp:
            return ValidationResult(is_valid=True, confidence=0.5)

        doc = self.spacy_nlp(value)

        # Check if spaCy recognizes it as a person
        has_person_entity = any(ent.label_ == "PERSON" for ent in doc.ents)

        if has_person_entity:
            return ValidationResult(
                is_valid=True,
                confidence=0.9,
                error_type=None
            )

        # Heuristic: Check if it looks like a name (1-4 capitalized words)
        words = value.split()
        is_capitalized = all(word[0].isupper() for word in words if word)
        reasonable_length = 1 <= len(words) <= 4

        if is_capitalized and reasonable_length:
            return ValidationResult(
                is_valid=True,
                confidence=0.6,
                error_type=None
            )

        return ValidationResult(
            is_valid=False,
            confidence=0.3,
            error_type="invalid_name_format"
        )

    def _validate_organization(self, value: str) -> ValidationResult:
        """
        Validate organization names using spaCy NER.

        Args:
            value: Organization name to validate

        Returns:
            ValidationResult
        """
        if not self.spacy_nlp:
            return ValidationResult(is_valid=True, confidence=0.5)

        doc = self.spacy_nlp(value)

        # Check if spaCy recognizes it as an organization
        has_org_entity = any(ent.label_ == "ORG" for ent in doc.ents)

        if has_org_entity:
            return ValidationResult(
                is_valid=True,
                confidence=0.9,
                error_type=None
            )

        # Organizations often have capitalized words or special suffixes
        has_suffix = any(
            suffix in value.upper()
            for suffix in ["INC", "LLC", "LTD", "CORP", "CO"]
        )

        if has_suffix:
            return ValidationResult(
                is_valid=True,
                confidence=0.7,
                error_type=None
            )

        # Assume valid but low confidence
        return ValidationResult(
            is_valid=True,
            confidence=0.5,
            error_type=None
        )

    def _validate_by_similarity(self, value: str,
                                column_samples: List[str]) -> ValidationResult:
        """
        Validate using semantic similarity to other values in the column.

        This is used when we don't know the data type. We check if the value
        is semantically similar to other values in the column.

        Args:
            value: Value to validate
            column_samples: Other values from the same column

        Returns:
            ValidationResult
        """
        if not self.sentence_model or not column_samples:
            return ValidationResult(is_valid=True, confidence=0.5)

        # Filter out empty samples
        valid_samples = [
            str(s).strip() for s in column_samples
            if str(s).strip() and str(s).strip().lower() not in ['nan', 'none', 'null']
        ]

        if not valid_samples or len(valid_samples) < 3:
            return ValidationResult(is_valid=True, confidence=0.5)

        try:
            # Encode value and samples
            embeddings = self.sentence_model.encode([value] + valid_samples)

            value_embedding = embeddings[0:1]
            sample_embeddings = embeddings[1:]

            # Calculate similarity
            similarities = cosine_similarity(value_embedding, sample_embeddings)[0]

            # Get statistics
            max_similarity = float(np.max(similarities))
            mean_similarity = float(np.mean(similarities))

            # Validation logic:
            # - If very similar to at least one sample (>0.7) -> valid
            # - If moderately similar to most samples (mean >0.5) -> valid
            # - Otherwise -> invalid

            if max_similarity > 0.7:
                is_valid = True
                confidence = max_similarity
            elif mean_similarity > 0.5:
                is_valid = True
                confidence = mean_similarity
            else:
                is_valid = False
                confidence = mean_similarity

            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                error_type="low_similarity" if not is_valid else None,
                metadata={
                    "max_similarity": max_similarity,
                    "mean_similarity": mean_similarity
                }
            )
        except Exception as e:
            # If embeddings fail, assume valid
            return ValidationResult(
                is_valid=True,
                confidence=0.5,
                error_type=None,
                metadata={"error": str(e)}
            )

    def validate_batch(self, values: List[str], column_name: str = "",
                      column_samples: List[str] = None) -> List[ValidationResult]:
        """
        Validate multiple values at once.

        Args:
            values: List of values to validate
            column_name: Name of the column
            column_samples: Sample values for context

        Returns:
            List of ValidationResult objects
        """
        if column_samples is None:
            column_samples = values[:min(20, len(values))]  # Use first 20 as samples

        return [
            self.validate(value, column_name, column_samples)
            for value in values
        ]

    def get_data_type(self) -> str:
        """Get the data type this validator handles."""
        return "text"  # Universal text validator

    def is_model_loaded(self) -> bool:
        """Check if models are loaded and ready."""
        return self.is_trained and (self.spacy_nlp is not None or self.sentence_model is not None)

    def load_model(self, filepath: str) -> bool:
        """
        Not needed for NLP validator (no pre-training required).

        Returns:
            True (always ready)
        """
        return True

    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            'data_type': self.get_data_type(),
            'is_loaded': self.is_model_loaded(),
            'model_path': None,
            'spacy_available': self.spacy_nlp is not None,
            'sentence_transformer_available': self.sentence_model is not None,
            'reference_databases': ['pycountry'],
            'supported_entities': ['location', 'person', 'organization', 'country']
        }


if __name__ == "__main__":
    # Test the validator
    print("Testing Universal NLP Validator...")
    print("=" * 60)

    validator = UniversalNLPValidator()

    # Test country validation
    print("\n1. Country Validation:")
    test_countries = ["USA", "United States", "Sinapore", "UK", "Germny"]
    for country in test_countries:
        result = validator.validate(country, "Country")
        print(f"  {country:<20} -> Valid: {result.is_valid}, Confidence: {result.confidence:.2f}")
        if result.metadata and 'best_match' in result.metadata:
            print(f"                         Suggestion: {result.metadata['best_match']}")

    # Test location validation
    print("\n2. Location Validation:")
    test_locations = ["New York", "Paris", "xyzabc", "London"]
    for location in test_locations:
        result = validator.validate(location, "City")
        print(f"  {location:<20} -> Valid: {result.is_valid}, Confidence: {result.confidence:.2f}")

    # Test person name validation
    print("\n3. Person Name Validation:")
    test_names = ["John Smith", "INVALID123", "Mary Johnson", "abc"]
    for name in test_names:
        result = validator.validate(name, "CustomerName")
        print(f"  {name:<20} -> Valid: {result.is_valid}, Confidence: {result.confidence:.2f}")

    # Test similarity-based validation
    print("\n4. Similarity-Based Validation:")
    samples = ["Apple", "Orange", "Banana", "Grape", "Mango"]
    test_values = ["Pineapple", "Car", "Strawberry", "Computer"]
    for value in test_values:
        result = validator.validate(value, "Fruit", samples)
        print(f"  {value:<20} -> Valid: {result.is_valid}, Confidence: {result.confidence:.2f}")

    print("\n" + "=" * 60)
    print("Model Info:")
    print(validator.get_model_info())
