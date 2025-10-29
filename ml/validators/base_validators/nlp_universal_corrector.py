"""
Universal NLP-Based Corrector

This corrector suggests corrections for ANY text data using NLP techniques:
1. Fuzzy matching for known entities (countries, locations)
2. Nearest neighbor search using semantic similarity
3. spaCy-based entity correction

Example usage:
    corrector = UniversalNLPCorrector()
    result = corrector.correct("Sinapore", "Country", ["USA", "UK", "Singapore"])
    # Returns: CorrectionResult(corrected_value="Singapore", confidence=0.85)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process
import pycountry
import spacy
from typing import List, Optional
import warnings

from ml.base_corrector import BaseCorrector, CorrectionResult


class UniversalNLPCorrector(BaseCorrector):
    """
    Universal corrector that suggests fixes for any text column.

    Uses NLP techniques to find the most likely correction:
    - Fuzzy matching for known reference data (countries, etc.)
    - Semantic similarity for unknown data types
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the universal NLP corrector."""
        super().__init__(model_path)

        try:
            self.spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            warnings.warn(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            self.spacy_nlp = None

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

        self.is_trained = True

    def correct(self, value: str, column_name: str = "",
                column_samples: List[str] = None) -> CorrectionResult:
        """
        Suggest correction for a single invalid value.

        Args:
            value: The invalid value to correct
            column_name: Name of the column (provides context)
            column_samples: Valid sample values from the same column

        Returns:
            CorrectionResult with suggested correction
        """
        if column_samples is None:
            column_samples = []

        value_str = str(value).strip()

        # Empty value - no correction possible
        if not value_str or value_str.lower() in ['nan', 'none', 'null', '']:
            return CorrectionResult(
                original_value=value,
                corrected_value=None,
                confidence=0.0,
                correction_type="no_correction"
            )

        # Detect entity type from column name
        entity_type = self._detect_entity_type(column_name, column_samples)

        # Apply appropriate correction technique
        if entity_type == "country":
            return self._correct_country(value_str)
        elif entity_type in ["location", "person", "organization"]:
            return self._correct_by_fuzzy_match(value_str, column_samples)
        else:
            # Unknown type - use similarity-based correction
            if column_samples and len(column_samples) >= 3:
                return self._correct_by_similarity(value_str, column_samples)
            else:
                # Not enough data to suggest correction
                return CorrectionResult(
                    original_value=value,
                    corrected_value=None,
                    confidence=0.0,
                    correction_type="insufficient_data"
                )

    def _detect_entity_type(self, column_name: str,
                           column_samples: List[str]) -> str:
        """Detect what type of entity this column contains."""
        column_lower = column_name.lower()

        if any(word in column_lower for word in ['country', 'nation', 'nationality']):
            return "country"

        if any(word in column_lower for word in ['city', 'location', 'place', 'state', 'region']):
            return "location"

        if any(word in column_lower for word in ['name', 'person', 'customer', 'employee']):
            return "person"

        if any(word in column_lower for word in ['company', 'organization', 'org', 'business']):
            return "organization"

        return "unknown"

    def _correct_country(self, value: str) -> CorrectionResult:
        """
        Correct country names using fuzzy matching against known countries.

        Args:
            value: Invalid country name

        Returns:
            CorrectionResult with suggested country
        """
        # Use fuzzy matching to find best match
        best_match, score = process.extractOne(
            value,
            self.countries,
            scorer=fuzz.ratio
        )

        confidence = score / 100.0

        # Only suggest if confidence is reasonable (>60%)
        if confidence > 0.6:
            return CorrectionResult(
                original_value=value,
                corrected_value=best_match,
                confidence=confidence,
                correction_type="fuzzy_match",
                metadata={"match_score": score}
            )
        else:
            return CorrectionResult(
                original_value=value,
                corrected_value=None,
                confidence=confidence,
                correction_type="low_confidence"
            )

    def _correct_by_fuzzy_match(self, value: str,
                                column_samples: List[str]) -> CorrectionResult:
        """
        Correct using fuzzy matching against sample values.

        Args:
            value: Invalid value
            column_samples: Valid values from the column

        Returns:
            CorrectionResult
        """
        if not column_samples:
            return CorrectionResult(
                original_value=value,
                corrected_value=None,
                confidence=0.0,
                correction_type="no_samples"
            )

        # Filter out empty samples
        valid_samples = [
            str(s).strip() for s in column_samples
            if str(s).strip() and str(s).strip().lower() not in ['nan', 'none', 'null']
        ]

        if not valid_samples:
            return CorrectionResult(
                original_value=value,
                corrected_value=None,
                confidence=0.0,
                correction_type="no_valid_samples"
            )

        # Find best fuzzy match
        best_match, score = process.extractOne(
            value,
            valid_samples,
            scorer=fuzz.ratio
        )

        confidence = score / 100.0

        # Threshold: only suggest if >70% similar
        if confidence > 0.7 and best_match.lower() != value.lower():
            return CorrectionResult(
                original_value=value,
                corrected_value=best_match,
                confidence=confidence,
                correction_type="fuzzy_match",
                metadata={"match_score": score}
            )
        else:
            return CorrectionResult(
                original_value=value,
                corrected_value=None,
                confidence=confidence,
                correction_type="no_good_match"
            )

    def _correct_by_similarity(self, value: str,
                              column_samples: List[str]) -> CorrectionResult:
        """
        Correct using semantic similarity (nearest neighbor in embedding space).

        Args:
            value: Invalid value
            column_samples: Valid values from the column

        Returns:
            CorrectionResult
        """
        if not self.sentence_model:
            # Fallback to fuzzy matching
            return self._correct_by_fuzzy_match(value, column_samples)

        # Filter out empty samples
        valid_samples = [
            str(s).strip() for s in column_samples
            if str(s).strip() and str(s).strip().lower() not in ['nan', 'none', 'null']
        ]

        if len(valid_samples) < 3:
            return CorrectionResult(
                original_value=value,
                corrected_value=None,
                confidence=0.0,
                correction_type="insufficient_samples"
            )

        try:
            # Encode value and samples
            embeddings = self.sentence_model.encode([value] + valid_samples)

            value_embedding = embeddings[0:1]
            sample_embeddings = embeddings[1:]

            # Find most similar sample
            similarities = cosine_similarity(value_embedding, sample_embeddings)[0]
            most_similar_idx = int(np.argmax(similarities))
            max_similarity = float(similarities[most_similar_idx])

            best_match = valid_samples[most_similar_idx]

            # Only suggest if reasonably similar (>0.5) and different
            if max_similarity > 0.5 and best_match.lower() != value.lower():
                return CorrectionResult(
                    original_value=value,
                    corrected_value=best_match,
                    confidence=max_similarity,
                    correction_type="semantic_similarity",
                    metadata={"similarity_score": max_similarity}
                )
            else:
                return CorrectionResult(
                    original_value=value,
                    corrected_value=None,
                    confidence=max_similarity,
                    correction_type="low_similarity"
                )

        except Exception as e:
            # If similarity fails, try fuzzy matching
            return self._correct_by_fuzzy_match(value, column_samples)

    def correct_batch(self, values: List[str], column_name: str = "",
                     column_samples: List[str] = None) -> List[CorrectionResult]:
        """
        Correct multiple values at once.

        Args:
            values: List of invalid values to correct
            column_name: Name of the column
            column_samples: Valid sample values for context

        Returns:
            List of CorrectionResult objects
        """
        if column_samples is None:
            # Use all values as samples (assuming most are valid)
            column_samples = values

        return [
            self.correct(value, column_name, column_samples)
            for value in values
        ]

    def get_data_type(self) -> str:
        """Get the data type this corrector handles."""
        return "text"  # Universal text corrector

    def is_model_loaded(self) -> bool:
        """Check if models are loaded and ready."""
        return self.is_trained

    def load_model(self, filepath: str) -> bool:
        """
        Not needed for NLP corrector (no pre-training required).

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
            'correction_methods': ['fuzzy_match', 'semantic_similarity']
        }


if __name__ == "__main__":
    # Test the corrector
    print("Testing Universal NLP Corrector...")
    print("=" * 60)

    corrector = UniversalNLPCorrector()

    # Test country correction
    print("\n1. Country Correction:")
    test_countries = [
        ("Sinapore", "Country", ["USA", "UK", "Singapore", "Germany"]),
        ("Germny", "Country", []),
        ("United Sates", "Country", []),
    ]

    for value, col_name, samples in test_countries:
        result = corrector.correct(value, col_name, samples)
        print(f"  {value:<20} -> {result.corrected_value or 'No suggestion'} "
              f"(confidence: {result.confidence:.2f})")

    # Test fuzzy correction with samples
    print("\n2. Location Correction (with samples):")
    location_samples = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    test_locations = ["New Yrok", "Chicgo", "Huston"]

    for location in test_locations:
        result = corrector.correct(location, "City", location_samples)
        print(f"  {location:<20} -> {result.corrected_value or 'No suggestion'} "
              f"(confidence: {result.confidence:.2f})")

    # Test semantic similarity correction
    print("\n3. Product Correction (semantic similarity):")
    product_samples = ["Laptop", "Desktop Computer", "Tablet", "Smartphone", "Monitor"]
    test_products = ["Labtop", "Smart Phone", "Moniter"]

    for product in test_products:
        result = corrector.correct(product, "Product", product_samples)
        print(f"  {product:<20} -> {result.corrected_value or 'No suggestion'} "
              f"(confidence: {result.confidence:.2f}, type: {result.correction_type})")

    print("\n" + "=" * 60)
    print("Model Info:")
    print(corrector.get_model_info())
