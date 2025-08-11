import phonenumbers
import re

def is_phone_number_valid(number_str):
    """Validate phone number using custom rules and phonenumbers library."""
    number_str = str(number_str).strip()

    # Rule 1: Must not have multiple '+' signs
    if number_str.count('+') > 1:
        return False

    # Rule 2: Only digits, spaces, dashes allowed (optional leading '+')
    if not re.match(r'^\+?\d[\d\s\-]*$', number_str):
        return False

    # Rule 3: Use phonenumbers library for deep validation
    try:
        parsed = phonenumbers.parse(number_str, None)
        return phonenumbers.is_valid_number(parsed)
    except phonenumbers.NumberParseException:
        return False
