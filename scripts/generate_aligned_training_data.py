"""
Generate ALIGNED training data for both models

This ensures:
1. Logistic Regression learns which phones are valid/invalid
2. XGBoost learns how to FIX the exact invalid phones that Logistic flags

CRITICAL: Both models must see the SAME phone numbers!
"""

import pandas as pd
import numpy as np
import os


def generate_aligned_training_data(num_samples=5000):
    """
    Generate training data where:
    - Valid phones are used for Logistic (label=1) but NOT for XGBoost
    - Invalid phones are used for BOTH:
      - Logistic (label=0)
      - XGBoost (invalid → valid correction)
    """
    np.random.seed(42)

    # Country codes
    countries = [
        ('+1', 10),   # US
        ('+44', 10),  # UK
        ('+61', 9),   # Australia
        ('+86', 11),  # China
        ('+91', 10),  # India
        ('+65', 8),   # Singapore
        ('+33', 9),   # France
    ]

    logistic_data = []
    xgboost_data = []

    print("Generating aligned training data...")
    print(f"Target: {num_samples} samples")

    for i in range(num_samples):
        # Generate a valid phone
        country_code, num_digits = countries[np.random.randint(0, len(countries))]
        digits = ''.join([str(np.random.randint(0, 10)) for _ in range(num_digits)])
        valid_phone = country_code + digits

        # Add to Logistic as VALID
        logistic_data.append({
            'phone': valid_phone,
            'is_valid': 1
        })

        # Now create an INVALID version and add to BOTH models
        corruption_type = np.random.choice([
            'missing_country',  # 40%
            'missing_country',
            'missing_country',
            'missing_country',
            'letter_typo',      # 30%
            'letter_typo',
            'letter_typo',
            'spaces',           # 20%
            'spaces',
            'mixed',            # 10%
        ])

        invalid_phone = valid_phone
        edit_ops = ['keep'] * len(valid_phone)

        if corruption_type == 'missing_country':
            # Remove country code: +1234567890 → 1234567890
            invalid_phone = digits
            edit_ops = ['insert_country'] + ['keep'] * len(digits)

        elif corruption_type == 'letter_typo':
            # Replace 1-3 digits with letters: +1234567890 → +12345678o0
            chars = list(valid_phone)
            ops = list(edit_ops)

            num_typos = np.random.randint(1, 4)
            for _ in range(num_typos):
                digit_positions = [i for i, c in enumerate(chars) if c.isdigit()]
                if digit_positions:
                    pos = np.random.choice(digit_positions)
                    original_digit = chars[pos]

                    # Map digit to similar letter
                    typo_map = {
                        '0': 'o', '1': 'l', '3': 'e',
                        '5': 's', '8': 'b', '9': 'g', '2': 'a'
                    }
                    if original_digit in typo_map:
                        chars[pos] = typo_map[original_digit]
                        ops[pos] = f'replace_{original_digit}'

            invalid_phone = ''.join(chars)
            edit_ops = ops

        elif corruption_type == 'spaces':
            # Add spaces: +1234567890 → +123 456 7890
            chars = list(valid_phone)
            ops = list(edit_ops)

            num_spaces = np.random.randint(1, 3)
            for _ in range(num_spaces):
                if len(chars) > 3:
                    pos = np.random.randint(2, len(chars))
                    chars.insert(pos, ' ')
                    ops.insert(pos, 'delete')

            invalid_phone = ''.join(chars)
            edit_ops = ops

        elif corruption_type == 'mixed':
            # Multiple issues: missing country + letter typo + spaces
            # Start without country code
            invalid_phone = digits
            chars = list(invalid_phone)

            # Add 1-2 letter typos
            num_typos = np.random.randint(1, 3)
            for _ in range(num_typos):
                if chars:
                    pos = np.random.randint(0, len(chars))
                    if chars[pos].isdigit():
                        typo_map = {'0': 'o', '1': 'l', '3': 'e', '5': 's'}
                        if chars[pos] in typo_map:
                            original = chars[pos]
                            chars[pos] = typo_map[chars[pos]]

            # Add spaces
            if len(chars) > 5:
                pos = np.random.randint(3, len(chars) - 2)
                chars.insert(pos, ' ')
                chars.insert(pos + 3, ' ')

            invalid_phone = ''.join(chars)

            # Edit ops: need to handle insert_country + replacements + delete spaces
            # This is complex, so simplified:
            edit_ops = ['insert_country'] + ['keep'] * len(digits)

        # Add invalid version to Logistic as INVALID
        logistic_data.append({
            'phone': invalid_phone,
            'is_valid': 0
        })

        # Add correction pair to XGBoost
        xgboost_data.append({
            'invalid': invalid_phone,
            'valid': valid_phone,
            'operations': '|'.join(edit_ops)
        })

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    logistic_df = pd.DataFrame(logistic_data)
    xgboost_df = pd.DataFrame(xgboost_data)

    print(f"\nLogistic Regression dataset:")
    print(f"  - Total samples: {len(logistic_df)}")
    print(f"  - Valid phones: {logistic_df['is_valid'].sum()}")
    print(f"  - Invalid phones: {len(logistic_df) - logistic_df['is_valid'].sum()}")

    print(f"\nXGBoost Corrector dataset:")
    print(f"  - Total correction pairs: {len(xgboost_df)}")

    return logistic_df, xgboost_df


if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING ALIGNED TRAINING DATA")
    print("=" * 70)
    print("\nThis ensures both models work on the SAME phone numbers!")
    print("- Logistic Regression: Classifies phones as valid/invalid")
    print("- XGBoost: Learns to FIX the invalid phones Logistic flags")
    print()

    # Create data directory
    data_dir = '../data'
    os.makedirs(data_dir, exist_ok=True)

    # Generate aligned data
    logistic_df, xgboost_df = generate_aligned_training_data(num_samples=5000)

    # Save files
    lr_path = os.path.join(data_dir, 'logistic_regression_training.csv')
    xgb_path = os.path.join(data_dir, 'xgboost_corrector_training.csv')

    logistic_df.to_csv(lr_path, index=False)
    xgboost_df.to_csv(xgb_path, index=False)

    print(f"\n✓ Saved Logistic Regression data to: {lr_path}")
    print(f"✓ Saved XGBoost Corrector data to: {xgb_path}")

    # Show examples of alignment
    print("\n" + "=" * 70)
    print("ALIGNMENT EXAMPLES")
    print("=" * 70)

    print("\nExample 1:")
    print(f"  Logistic sees: '{xgboost_df.iloc[0]['invalid']}' → INVALID (label=0)")
    print(f"  XGBoost fixes: '{xgboost_df.iloc[0]['invalid']}' → '{xgboost_df.iloc[0]['valid']}'")

    print("\nExample 2:")
    print(f"  Logistic sees: '{xgboost_df.iloc[1]['invalid']}' → INVALID (label=0)")
    print(f"  XGBoost fixes: '{xgboost_df.iloc[1]['invalid']}' → '{xgboost_df.iloc[1]['valid']}'")

    print("\nExample 3:")
    print(f"  Logistic sees: '{xgboost_df.iloc[2]['invalid']}' → INVALID (label=0)")
    print(f"  XGBoost fixes: '{xgboost_df.iloc[2]['invalid']}' → '{xgboost_df.iloc[2]['valid']}'")

    print("\n" + "=" * 70)
    print("ALIGNED TRAINING DATA GENERATION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run app: streamlit run app.py")
    print("2. Go to Model Training tab")
    print("3. Train both models using the new aligned data")
    print("4. Both models will now work on the SAME phone numbers!")
