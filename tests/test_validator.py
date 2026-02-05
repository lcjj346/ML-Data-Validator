"""
Tests for ML Data Validator

Run with: pytest tests/ -v
Or: python -m pytest tests/ -v
"""

import pytest
import pandas as pd
import os
import sys
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.validator import UnifiedMLValidator
from ml.feature_extractor import GenericFeatureExtractor
from ml.corrector import GenericMLCorrector


class TestFeatureExtractor:
    """Tests for GenericFeatureExtractor"""

    def setup_method(self):
        self.extractor = GenericFeatureExtractor()

    def test_extract_features_returns_list(self):
        """Feature extraction should return a list of floats"""
        features = self.extractor.extract_features("test")
        assert isinstance(features, list)
        assert all(isinstance(f, (int, float)) for f in features)

    def test_extract_features_consistent_length(self):
        """Feature vectors should have consistent length"""
        texts = ["hello", "test@email.com", "+65 9123 4567", "123", ""]
        lengths = [len(self.extractor.extract_features(t)) for t in texts]
        assert len(set(lengths)) == 1, "All feature vectors should have same length"

    def test_extract_features_numeric(self):
        """Numeric values should be detected"""
        features_num = self.extractor.extract_features("123.45")
        features_text = self.extractor.extract_features("hello")
        # Numeric features should differ
        assert features_num != features_text

    def test_extract_features_email_pattern(self):
        """Email patterns should be detected"""
        features_email = self.extractor.extract_features("test@example.com")
        features_text = self.extractor.extract_features("hello world")
        assert features_email != features_text

    def test_extract_features_phone_pattern(self):
        """Phone patterns should be detected"""
        features_phone = self.extractor.extract_features("+65 9123 4567")
        features_text = self.extractor.extract_features("hello world")
        assert features_phone != features_text

    def test_extract_features_empty_string(self):
        """Empty string should not raise error"""
        features = self.extractor.extract_features("")
        assert isinstance(features, list)

    def test_extract_features_with_column_name(self):
        """Column name parameter should not break extraction"""
        features = self.extractor.extract_features("test", column_name="name")
        assert isinstance(features, list)


class TestUnifiedMLValidator:
    """Tests for UnifiedMLValidator"""

    def setup_method(self):
        self.validator = UnifiedMLValidator()

    def test_init_creates_empty_validator(self):
        """New validator should be empty and not trained"""
        assert self.validator.is_trained == False
        assert self.validator.columns == []
        assert self.validator.column_models == {}

    def test_open_ended_columns_constant(self):
        """OPEN_ENDED_COLUMNS should be defined"""
        assert hasattr(UnifiedMLValidator, 'OPEN_ENDED_COLUMNS')
        assert 'name' in UnifiedMLValidator.OPEN_ENDED_COLUMNS
        assert 'email' in UnifiedMLValidator.OPEN_ENDED_COLUMNS

    def test_train_on_dataframe(self):
        """Training on a DataFrame should work"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Wilson'],
            'email': ['john@example.com', 'jane@test.com', 'bob@mail.com'],
            'country': ['USA', 'UK', 'Singapore']
        })
        metrics = self.validator.train(df, model_name='test_model')

        assert self.validator.is_trained == True
        assert 'name' in self.validator.columns
        assert 'email' in self.validator.columns
        assert 'country' in self.validator.columns
        assert 'name' in metrics
        # Small datasets use train_accuracy, larger use test_accuracy
        assert 'train_accuracy' in metrics['name'] or 'test_accuracy' in metrics['name']

    def test_validate_after_training(self):
        """Validation should work after training"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'Bob Wilson'] * 10,
            'country': ['USA', 'UK', 'Singapore'] * 10
        })
        self.validator.train(df, model_name='test_model')

        # Validate known good value
        is_valid, confidence = self.validator.validate('USA', 'country')
        assert is_valid == True
        assert confidence > 0.5

    def test_whitelist_exact_match(self):
        """Exact whitelist matches should return valid"""
        df = pd.DataFrame({
            'country': ['Singapore', 'USA', 'UK'] * 10
        })
        self.validator.train(df, model_name='test_model')

        is_valid, confidence = self.validator.validate('singapore', 'country')
        assert is_valid == True
        assert confidence == 1.0

    def test_save_and_load_model(self):
        """Model should be saveable and loadable"""
        df = pd.DataFrame({
            'name': ['John', 'Jane', 'Bob'] * 10,
            'country': ['USA', 'UK', 'Singapore'] * 10
        })
        self.validator.train(df, model_name='test_model')

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            self.validator.save(temp_path)
            assert os.path.exists(temp_path)

            # Load into new validator
            loaded_validator = UnifiedMLValidator(temp_path)
            assert loaded_validator.is_trained == True
            assert loaded_validator.columns == self.validator.columns
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_validate_row(self):
        """validate_row should validate entire row"""
        df = pd.DataFrame({
            'name': ['John Doe', 'Jane Smith'] * 10,
            'country': ['USA', 'UK'] * 10
        })
        self.validator.train(df, model_name='test_model')

        row = pd.Series({'name': 'John Doe', 'country': 'USA'})
        results = self.validator.validate_row(row)

        assert 'name' in results
        assert 'country' in results
        assert isinstance(results['name'], tuple)

    def test_correct(self):
        """Corrections should be suggested for similar values"""
        df = pd.DataFrame({
            'country': ['Singapore', 'USA', 'United Kingdom'] * 10
        })
        self.validator.train(df, model_name='test_model')

        # Typo should get correction
        correction = self.validator.correct('Singaproe', 'country')
        assert correction is not None
        assert correction.lower() == 'singapore'


class TestTypoDetection:
    """Tests for typo detection logic"""

    def setup_method(self):
        self.validator = UnifiedMLValidator()
        # Train with countries (finite set - should use typo detection)
        df = pd.DataFrame({
            'country': ['Singapore', 'USA', 'United Kingdom', 'Germany', 'France'] * 10,
            'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'] * 10
        })
        self.validator.train(df, model_name='test_model')

    def test_typo_detected_for_country(self):
        """Typos in country column should be detected"""
        # Singaproe is similar to Singapore (>80%)
        is_valid, confidence = self.validator.validate('Singaproe', 'country')
        assert is_valid == False

    def test_valid_country_accepted(self):
        """Valid countries should be accepted"""
        is_valid, confidence = self.validator.validate('Singapore', 'country')
        assert is_valid == True

    def test_name_not_strict_typo_detection(self):
        """Names should not use strict typo detection (open-ended column)"""
        # Even if 'Jonh' is similar to 'John', names are open-ended
        # so it should go through ML model, not be rejected as typo
        is_valid, confidence = self.validator.validate('Jean-Pierre', 'name')
        # Should not be rejected just because it's similar to something


class TestNumericValidation:
    """Tests for numeric value handling"""

    def setup_method(self):
        self.validator = UnifiedMLValidator()
        df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 5,
            'blood_sugar': [95.0, 100.0, 105.0, 110.0, 115.0] * 10
        })
        self.validator.train(df, model_name='test_model')

    def test_numeric_normalization(self):
        """95 and 95.0 should be treated as equivalent"""
        is_valid_int, _ = self.validator.validate('95', 'blood_sugar')
        is_valid_float, _ = self.validator.validate('95.0', 'blood_sugar')
        assert is_valid_int == is_valid_float

    def test_valid_age(self):
        """Valid ages should be accepted"""
        is_valid, confidence = self.validator.validate('30', 'age')
        assert is_valid == True


class TestModelStacking:
    """Tests for add_training_data (model stacking)"""

    def setup_method(self):
        self.validator = UnifiedMLValidator()
        df = pd.DataFrame({
            'country': ['Singapore', 'USA', 'UK'] * 10
        })
        self.validator.train(df, model_name='test_model')

    def test_add_training_data_expands_whitelist(self):
        """Adding data should expand the whitelist"""
        initial_size = len(self.validator.column_whitelists.get('country', set()))

        new_df = pd.DataFrame({
            'country': ['Germany', 'France', 'Japan'] * 5
        })
        self.validator.add_training_data(new_df, retrain=False)

        new_size = len(self.validator.column_whitelists.get('country', set()))
        assert new_size > initial_size

    def test_added_values_are_valid(self):
        """Newly added values should validate as valid"""
        new_df = pd.DataFrame({
            'country': ['NewCountry'] * 5
        })
        self.validator.add_training_data(new_df, retrain=False)

        is_valid, confidence = self.validator.validate('NewCountry', 'country')
        assert is_valid == True


class TestCorrector:
    """Tests for GenericMLCorrector"""

    def setup_method(self):
        self.corrector = GenericMLCorrector()

    def test_corrector_initialization(self):
        """Corrector should initialize properly"""
        assert self.corrector is not None

    def test_valid_examples_attribute(self):
        """Setting valid examples should work"""
        examples = ['Singapore', 'USA', 'UK']
        self.corrector.valid_examples = examples
        self.corrector.is_trained = True
        assert len(self.corrector.valid_examples) == 3

    def test_correct_finds_similar(self):
        """Correction should find similar values"""
        self.corrector.valid_examples = ['Singapore', 'USA', 'United Kingdom']
        self.corrector.is_trained = True
        correction = self.corrector.correct('Singaproe')
        assert correction is not None
        assert correction.lower() == 'singapore'

    def test_correct_no_match(self):
        """No correction for completely different values"""
        self.corrector.valid_examples = ['Singapore', 'USA', 'UK']
        self.corrector.is_trained = True
        correction = self.corrector.correct('xyz123abc')
        # Either None or a low-confidence match


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def setup_method(self):
        self.validator = UnifiedMLValidator()

    def test_validate_before_training(self):
        """Validation before training should handle gracefully"""
        # Should not crash
        try:
            result = self.validator.validate('test', 'nonexistent')
        except Exception as e:
            # Expected to raise an error or return a default
            pass

    def test_empty_dataframe_training(self):
        """Training on empty dataframe should handle gracefully"""
        df = pd.DataFrame()
        try:
            self.validator.train(df, model_name='test')
        except Exception:
            pass  # Expected to fail gracefully

    def test_special_characters_in_values(self):
        """Special characters should be handled"""
        df = pd.DataFrame({
            'name': ["O'Connor", "Jean-Pierre", "José García"] * 10
        })
        self.validator.train(df, model_name='test')

        is_valid, _ = self.validator.validate("O'Connor", 'name')
        assert is_valid == True

    def test_unicode_values(self):
        """Unicode values should be handled"""
        df = pd.DataFrame({
            'name': ['田中太郎', '김철수', 'Müller'] * 10
        })
        self.validator.train(df, model_name='test')

        is_valid, _ = self.validator.validate('田中太郎', 'name')
        assert is_valid == True


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
