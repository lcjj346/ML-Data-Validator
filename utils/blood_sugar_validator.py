def is_blood_sugar_valid(blood_sugar) -> bool:
    """
    Validates blood sugar level based on a reasonable range.
    Typical fasting blood sugar (mg/dL): 70–100 (normal), 101–125 (pre-diabetes), >126 (diabetes).
    Typical mmol/L range: ~3.9–7.0.

    Accept a reasonable extended range to avoid over-rejecting unusual but possible values.
    """
    try:
        if blood_sugar is None or str(blood_sugar).strip() == "":
            return False

        blood_sugar = float(blood_sugar)

        # Check for negative or unrealistic values
        if blood_sugar <= 0:
            return False

        # Acceptable range: 50 to 500 mg/dL (flexible for both fasting & random tests)
        return 50 <= blood_sugar <= 500

    except ValueError:
        return False
