"""
Email Validator

Pattern-based and ML-enhanced email address validator.
Uses regex patterns and optional ML for advanced validation.

Example usage:
    from ml.validators.email_validator import EmailValidator

    validator = EmailValidator()
    result = validator.validate('user@example.com')
    print(result.is_valid, result.confidence)
"""

import re
from typing import Any, List, Optional
from ml.base_validator import BaseValidator, ValidationResult


class EmailValidator(BaseValidator):
    """
    Email address validator using pattern matching.

    Validates email addresses using:
    - RFC-compliant regex patterns
    - Domain validation
    - Common typo detection

    This is a rule-based validator (no ML training required).
    """

    # Comprehensive email regex pattern
    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$'
    )

    # Common valid domains for confidence boosting
    COMMON_DOMAINS = {
        'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
        'icloud.com', 'aol.com', 'protonmail.com', 'mail.com'
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the email validator.

        Args:
            model_path: Not used (rule-based validator)
        """
        super().__init__(model_path)
        self.is_trained = True  # Always ready

    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a single email address.

        Args:
            value: Email address to validate

        Returns:
            ValidationResult with validation outcome
        """
        email_str = str(value).strip() if value is not None else ""

        # Check if empty
        if not email_str:
            return ValidationResult(
                is_valid=False,
                confidence=1.0,
                error_type="empty_value",
                metadata={'original_value': email_str}
            )

        # Check basic pattern
        if not self.EMAIL_PATTERN.match(email_str):
            return ValidationResult(
                is_valid=False,
                confidence=0.9,
                error_type="format_error",
                metadata={'original_value': email_str, 'reason': 'invalid_pattern'}
            )

        # Extract domain
        try:
            _, domain = email_str.rsplit('@', 1)
        except ValueError:
            return ValidationResult(
                is_valid=False,
                confidence=0.95,
                error_type="format_error",
                metadata={'original_value': email_str, 'reason': 'missing_@'}
            )

        # Additional validation checks
        confidence = 0.7  # Base confidence for valid pattern

        # Boost confidence for common domains
        if domain.lower() in self.COMMON_DOMAINS:
            confidence = 0.95

        # Check for suspicious patterns
        if '..' in email_str:
            return ValidationResult(
                is_valid=False,
                confidence=0.9,
                error_type="format_error",
                metadata={'original_value': email_str, 'reason': 'consecutive_dots'}
            )

        # Check domain has valid TLD
        if '.' not in domain:
            return ValidationResult(
                is_valid=False,
                confidence=0.9,
                error_type="format_error",
                metadata={'original_value': email_str, 'reason': 'no_tld'}
            )

        # Check for common typos in domain
        typos_detected = self._check_domain_typos(domain)
        if typos_detected:
            confidence *= 0.8  # Reduce confidence if typos suspected
            metadata = {
                'original_value': email_str,
                'possible_typos': typos_detected
            }
        else:
            metadata = {'original_value': email_str, 'domain': domain}

        return ValidationResult(
            is_valid=True,
            confidence=confidence,
            error_type=None,
            metadata=metadata
        )

    def _check_domain_typos(self, domain: str) -> List[str]:
        """
        Check for common typos in email domains.

        Args:
            domain: Email domain to check

        Returns:
            List of suspected typos
        """
        typos = []

        # Common typos
        typo_map = {
            'gmial.com': 'gmail.com',
            'gmai.com': 'gmail.com',
            'yahooo.com': 'yahoo.com',
            'yaho.com': 'yahoo.com',
            'outlok.com': 'outlook.com',
            'hotmial.com': 'hotmail.com',
            'homail.com': 'hotmail.com',  # Added
            'hotmil.com': 'hotmail.com',  # Added
        }

        domain_lower = domain.lower()
        if domain_lower in typo_map:
            typos.append(f"Possible typo: did you mean {typo_map[domain_lower]}?")

        return typos

    def validate_batch(self, values: List[Any]) -> List[ValidationResult]:
        """
        Validate multiple email addresses at once.

        Args:
            values: List of email addresses to validate

        Returns:
            List of ValidationResult objects
        """
        return super().validate_batch(values)

    def get_data_type(self) -> str:
        """
        Get the data type this validator handles.

        Returns:
            'email'
        """
        return 'email'

    def is_model_loaded(self) -> bool:
        """
        Check if validator is ready.

        Returns:
            Always True (rule-based validator)
        """
        return True

    def load_model(self, filepath: str) -> bool:
        """
        Load model (not applicable for rule-based validator).

        Args:
            filepath: Model file path

        Returns:
            False (not applicable)
        """
        return False

    def get_model_info(self) -> dict:
        """
        Get information about the email validator.

        Returns:
            Dictionary with validator metadata
        """
        return {
            'data_type': self.get_data_type(),
            'is_loaded': True,
            'model_type': 'Pattern-based',
            'description': 'RFC-compliant email validation with typo detection'
        }


if __name__ == "__main__":
    # Test the email validator
    print("Testing Email Validator")
    print("=" * 60)

    validator = EmailValidator()

    test_emails = [
        'user@example.com',
        'test.user@domain.co.uk',
        'invalid.email',
        'user@',
        '@example.com',
        'user..name@example.com',
        'user@gmial.com',  # Typo
        '',
        'admin@company.com',
    ]

    for email in test_emails:
        result = validator.validate(email)
        print(f"Email: {email:<30} -> Valid: {result.is_valid}, "
              f"Confidence: {result.confidence:.2f}, Error: {result.error_type}")
