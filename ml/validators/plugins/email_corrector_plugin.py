"""
Email Corrector Plugin

Suggests corrections for email addresses with typos in the domain.
Uses fuzzy matching to detect domain typos (e.g., homail.com → hotmail.com).

Example usage:
    from ml.validators.email_corrector_plugin import EmailCorrectorPlugin

    corrector = EmailCorrectorPlugin()
    result = corrector.correct('user@homail.com')
    print(result.corrected_value)  # user@hotmail.com
"""

from typing import Optional, Any, List
from fuzzywuzzy import fuzz, process
from ml.base_corrector import BaseCorrector, CorrectionResult


class EmailCorrectorPlugin(BaseCorrector):
    """
    Email corrector that detects and fixes domain typos using fuzzy matching.

    Works by:
    1. Extracting the domain from email
    2. Fuzzy matching against common domains
    3. Suggesting correction if similarity > threshold
    """

    # Common email domains to match against
    COMMON_DOMAINS = [
        'gmail.com',
        'yahoo.com',
        'outlook.com',
        'hotmail.com',
        'icloud.com',
        'aol.com',
        'protonmail.com',
        'mail.com',
        'live.com',
        'msn.com',
        'ymail.com',
        'googlemail.com',
    ]

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the email corrector."""
        super().__init__(model_path)
        self.is_trained = True  # Always ready (rule-based)

    def correct(self, value: Any) -> CorrectionResult:
        """
        Suggest correction for an email address.

        Args:
            value: Email address to correct

        Returns:
            CorrectionResult with suggested correction
        """
        email_str = str(value).strip() if value is not None else ""

        # Empty or clearly invalid
        if not email_str or '@' not in email_str:
            return CorrectionResult(
                original_value=value,
                corrected_value=None,
                confidence=0.0,
                correction_type="no_correction"
            )

        # Extract domain
        try:
            local_part, domain = email_str.rsplit('@', 1)
        except ValueError:
            return CorrectionResult(
                original_value=value,
                corrected_value=None,
                confidence=0.0,
                correction_type="invalid_format"
            )

        # Check if domain is already valid (exact match)
        if domain.lower() in [d.lower() for d in self.COMMON_DOMAINS]:
            return CorrectionResult(
                original_value=value,
                corrected_value=None,
                confidence=1.0,
                correction_type="already_valid"
            )

        # Fuzzy match against common domains
        best_match, score = process.extractOne(
            domain.lower(),
            [d.lower() for d in self.COMMON_DOMAINS],
            scorer=fuzz.ratio
        )

        confidence = score / 100.0

        # Only suggest if similarity is high enough (> 70%)
        # but not exact match (< 100%)
        if 0.70 < confidence < 1.0:
            corrected_email = f"{local_part}@{best_match}"
            return CorrectionResult(
                original_value=value,
                corrected_value=corrected_email,
                confidence=confidence,
                correction_type="domain_typo",
                metadata={
                    'original_domain': domain,
                    'suggested_domain': best_match,
                    'similarity_score': score
                }
            )
        else:
            # Either too different or already correct
            return CorrectionResult(
                original_value=value,
                corrected_value=None,
                confidence=confidence,
                correction_type="low_confidence" if confidence < 0.7 else "already_valid"
            )

    def correct_batch(self, values: List[Any]) -> List[CorrectionResult]:
        """
        Correct multiple email addresses.

        Args:
            values: List of email addresses

        Returns:
            List of CorrectionResult objects
        """
        return [self.correct(value) for value in values]

    def get_data_type(self) -> str:
        """Get the data type this corrector handles."""
        return 'email'

    def is_model_loaded(self) -> bool:
        """Check if corrector is ready."""
        return True

    def load_model(self, filepath: str) -> bool:
        """Load model (not applicable for rule-based corrector)."""
        return False

    def get_model_info(self) -> dict:
        """Get information about the corrector."""
        return {
            'data_type': self.get_data_type(),
            'is_loaded': True,
            'model_type': 'Fuzzy matching',
            'description': 'Email domain typo correction using fuzzy matching',
            'known_domains': len(self.COMMON_DOMAINS)
        }


if __name__ == "__main__":
    # Test the email corrector
    print("Testing Email Corrector")
    print("=" * 60)

    corrector = EmailCorrectorPlugin()

    test_emails = [
        'user@homail.com',      # homail → hotmail
        'test@gmial.com',       # gmial → gmail
        'admin@yahooo.com',     # yahooo → yahoo
        'user@outlok.com',      # outlok → outlook
        'valid@gmail.com',      # Already valid
        'user@unknown.com',     # Unknown domain (too different)
    ]

    for email in test_emails:
        result = corrector.correct(email)
        if result.corrected_value:
            print(f"{email:<25} → {result.corrected_value:<25} ({result.confidence:.0%})")
        else:
            print(f"{email:<25} → No suggestion ({result.correction_type})")

    print("\n" + "=" * 60)
    print("Model Info:")
    print(corrector.get_model_info())
