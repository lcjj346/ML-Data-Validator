"""
Phone Corrector Plugin

Adapter that wraps the existing EditDistanceCorrector class to conform to the
BaseCorrector interface, making it compatible with the plugin architecture.
"""

from typing import Any, List, Optional
from ml.base_corrector import BaseCorrector, CorrectionResult
from ml.edit_distance_corrector import EditDistanceCorrector


class PhoneCorrectorPlugin(BaseCorrector):
    """
    Phone number corrector plugin that wraps the existing EditDistanceCorrector.

    This adapter allows the existing phone corrector to work with the
    new plugin-based architecture.
    """

    def __init__(self, model_path: Optional[str] = 'saved_models/edit_distance_corrector.pkl'):
        """
        Initialize the phone corrector plugin.

        Args:
            model_path: Path to the trained model file
        """
        super().__init__(model_path)
        self._corrector = EditDistanceCorrector()

        # Try to load the model if path exists
        if model_path:
            self.is_trained = self._corrector.load(model_path)
        else:
            self.is_trained = False

    def correct(self, value: Any) -> CorrectionResult:
        """
        Correct a single invalid phone number.

        Args:
            value: Invalid phone number to correct

        Returns:
            CorrectionResult with correction outcome

        Raises:
            ValueError: If model is not loaded
        """
        if not self.is_model_loaded():
            raise ValueError("Phone correction model is not loaded")

        phone_str = str(value) if value is not None else ""

        try:
            corrected = self._corrector.correct_phone(phone_str)

            if corrected and corrected != phone_str:
                return CorrectionResult(
                    original_value=phone_str,
                    corrected_value=corrected,
                    confidence=0.8,  # XGBoost corrections are fairly confident
                    correction_type="ml_correction",
                    metadata={'method': 'XGBoost edit distance'}
                )
            else:
                # Correction failed or no correction needed
                return CorrectionResult(
                    original_value=phone_str,
                    corrected_value=None,
                    confidence=0.0,
                    correction_type=None,
                    metadata={'reason': 'no_correction_possible'}
                )

        except Exception as e:
            return CorrectionResult(
                original_value=phone_str,
                corrected_value=None,
                confidence=0.0,
                correction_type=None,
                metadata={'error': str(e)}
            )

    def correct_batch(self, values: List[Any]) -> List[CorrectionResult]:
        """
        Correct multiple invalid phone numbers at once.

        Args:
            values: List of invalid phone numbers to correct

        Returns:
            List of CorrectionResult objects

        Raises:
            ValueError: If model is not loaded
        """
        if not self.is_model_loaded():
            raise ValueError("Phone correction model is not loaded")

        results = []
        for value in values:
            results.append(self.correct(value))

        return results

    def get_data_type(self) -> str:
        """
        Get the data type this corrector handles.

        Returns:
            'phone'
        """
        return 'phone'

    def is_model_loaded(self) -> bool:
        """
        Check if the phone correction model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self._corrector.is_loaded()

    def load_model(self, filepath: str) -> bool:
        """
        Load a pre-trained phone correction model.

        Args:
            filepath: Path to the model file

        Returns:
            True if successful, False otherwise
        """
        success = self._corrector.load(filepath)
        if success:
            self.model_path = filepath
            self.is_trained = True
        return success

    def get_model_info(self) -> dict:
        """
        Get information about the phone correction model.

        Returns:
            Dictionary with model metadata
        """
        return {
            'data_type': self.get_data_type(),
            'is_loaded': self.is_model_loaded(),
            'model_path': self.model_path,
            'model_type': 'XGBoost',
            'description': 'ML-based phone number correction using XGBoost edit distance'
        }
