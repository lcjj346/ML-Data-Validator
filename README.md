# ML Data Validator

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

### 3. Train Your First Model

**Tab: Train Models**

1. Prepare a CSV file with 2 columns:
   - `text`: Your data (phone numbers, emails, names, etc.)
   - `label`: Either "valid" or "invalid"

Example CSV (`training_data/phone_training.csv`):

```csv
text,label
+6591234567,valid
+65 9123 4567,valid
123,invalid
abc,invalid
```

2. Upload the CSV in the "Train Models" tab
3. Give your model a name (e.g., "phone", "email", "name")
4. Click "Train Validator & Corrector"
5. Done! Your model is saved to `models/`

### 4. Validate Your Data

**Tab: Validate Data**

1. Upload a CSV file with data to validate
2. Map columns to trained validators (auto-detected where possible)
3. Click "Validate"
4. View results:
   - **Green cells** = Valid
   - **Red cells** = Invalid with correction suggestions
   - **Orange cells** = Manually edited by you
   - Editable grid (click any cell to modify)
5. Review per-column summary cards and correction suggestions
6. Apply corrections individually or all at once
7. Export validated results as CSV

## Architecture

### Project Structure

```
ml-data-validator/
├── app.py                          # Main Streamlit application (~775 lines)
├── requirements.txt                # Python dependencies (7 packages)
├── README.md                       # This file
│
├── ml/                             # Core ML package (4 files)
│   ├── __init__.py                 # Package exports
│   ├── feature_extractor.py        # Extract 71 features from any text (~287 lines)
│   ├── validator.py                # Generic ML validator with whitelist (~437 lines)
│   └── corrector.py                # Similarity-based corrector (~244 lines)
│
├── models/                         # Trained models (18 .pkl files)
│   ├── phone_validator.pkl
│   ├── phone_corrector.pkl
│   ├── email_validator.pkl
│   ├── email_corrector.pkl
│   ├── name_validator.pkl
│   ├── name_corrector.pkl
│   ├── country_validator.pkl
│   ├── country_corrector.pkl
│   ├── address_validator.pkl
│   ├── address_corrector.pkl
│   ├── age_validator.pkl
│   ├── age_corrector.pkl
│   ├── blood_sugar_validator.pkl
│   ├── blood_sugar_corrector.pkl
│   ├── custom_company_validator.pkl
│   ├── custom_company_corrector.pkl
│   ├── custom_validator.pkl
│   └── custom_corrector.pkl
│
├── training_data/                  # Training datasets (8 CSV files)
│   ├── phone_training.csv          # 565 examples
│   ├── email_training.csv          # 229 examples
│   ├── name_training.csv           # 280 examples
│   ├── country_training.csv        # 70 examples
│   ├── address_training.csv        # 100 examples
│   ├── age_training.csv            # 98 examples
│   ├── blood_sugar_training.csv    # 132 examples
│   └── custom_company_training.csv # 100 examples
│
└── test_data/                      # Sample test data
    ├── sample.csv                  # Multi-column test file (phone, country, email, name)
    └── custom_company.csv          # Company-specific test data
```

### How It Works

#### 1. Feature Extraction (`feature_extractor.py`)

Extracts **71 generic features** from ANY text, grouped into 9 categories:

| Category | Count | Features |
|----------|-------|----------|
| **Length** | 3 | total length, word count, comma-separated parts |
| **Character Type Ratios** | 5 | digit%, letter%, space%, uppercase%, lowercase% |
| **Special Character Counts** | 9 | +, @, ., #, -, \_, (, ), / |
| **Position** | 5 | starts with uppercase/digit/+, ends with digit/letter |
| **Email Patterns** | 6 | email-like, exactly one @, has username, domain has dot, domain extension, common domain |
| **Phone & Mixed Patterns** | 2 | phone-like, mixed alphanumeric |
| **Regex Patterns** | 3 | digit sequences, capitalized words, long digit sequences |
| **Common Keywords** | 12 | blk, ave, road, street, singapore, com, net, org, edu, @, +, # |
| **Character N-grams** | 6 | repeated chars, triple chars, max bigram freq, char variety, vowel ratio, max consecutive consonants |
| **Numeric Value** | 20 | numeric value, is_numeric, squared, cubed, sqrt, log, inverse, sign, abs, is_negative, 9 range buckets, decimal places |

**Key Insight**: The SAME feature extractor works for phone numbers, emails, names, or ANY custom format.

#### 2. Validation (`validator.py`)

- **Algorithm**: Logistic Regression (scikit-learn)
- **Hybrid approach**: Whitelist exact-match (confidence 1.0) + ML classification for unknown values
- **Training**: Learns from YOUR examples (text, label pairs)
- **Cross-validation**: Automatic k-fold CV when dataset is large enough (10+ samples, 3+ per class)
- **Output**: `(is_valid: bool, confidence: float)`
- **Parameters**: `max_iter=5000`, `class_weight='balanced'`, `solver='lbfgs'`
- **Warnings**: Alerts for class imbalance (<20% or >80% of one class) and small datasets (<50 examples)
- **Storage**: Saves/loads models as .pkl files using joblib

#### 3. Correction (`corrector.py`)

- **Algorithm**: Similarity-based matching using difflib's SequenceMatcher
- **Method**: Finds most similar valid example from training data
- **Ranking**: Candidates sorted by similarity, then canonical form preference (e.g., "USA" preferred over "usa")
- **Threshold**: 0.6 (configurable)
- **Output**: Suggested correction or None if no good match

#### 4. UI (`app.py`)

- **Framework**: Streamlit with streamlit-aggrid for editable grids
- **Two Tabs**:
  - **Validate Data**: Upload CSV, map columns, validate, apply corrections, export
  - **Train Models**: Upload training data, train validator + corrector, view all models
- **Features**:
  - Collapsible sections with step badges
  - Cell-level validation with color-coded grid (green/red/orange)
  - Per-column summary cards with quality bars
  - Individual and bulk correction application
  - Toast notifications on corrections
  - Quality metrics dashboard
  - Progress bars during validation and training
  - CSV export of validated data

## Pre-trained Models & Data

### Included Training Data (`training_data/`)

| Dataset | Examples | Description |
|---------|----------|-------------|
| `phone_training.csv` | 565 | International phone formats (+65, +1, +44, etc.) |
| `email_training.csv` | 229 | Standard emails + domain typo variants |
| `name_training.csv` | 280 | Full names + format/content invalids |
| `country_training.csv` | 70 | Country names with case variations |
| `address_training.csv` | 100 | Street addresses with ZIP codes |
| `age_training.csv` | 98 | Numeric ages (0-120 range) |
| `blood_sugar_training.csv` | 132 | Blood glucose in mmol/L |
| `custom_company_training.csv` | 100 | Company name validation |

**Total**: ~2,047 training examples across 8 datasets

### Pre-trained Models (`models/`)

9 validator-corrector pairs (18 .pkl files):

1. **phone** - Phone number validation & correction
2. **email** - Email address validation & correction
3. **name** - Person name validation & correction
4. **country** - Country name validation & correction
5. **address** - Street address validation & correction
6. **age** - Age validation & correction
7. **blood_sugar** - Blood glucose reading validation & correction
8. **custom_company** - Company name validation & correction
9. **custom** - General custom data validation & correction

### Sample Test Data (`test_data/`)

- `sample.csv` - Multi-column test file with phones, countries, emails, and names (with intentional errors)
- `custom_company.csv` - Sample company-specific data for testing custom validators

## Key Features

### 1. Generic ML Pipeline

- **No hardcoded rules** - Pure feature-based learning
- **No domain-specific logic** - Works for ANY data type
- **Train on YOUR data** - Learns YOUR specific patterns
- **71 generic features** - Extracted from any text
- **Hybrid validation** - Whitelist for known values + ML for unknowns

### 2. Smart Correction System

- **Similarity-based** - Finds most similar valid examples
- **Canonical form preference** - Prefers "USA" over "usa", "Singapore" over "SINGAPORE"
- **Configurable threshold** - Adjust sensitivity (default: 0.6)
- **Fast suggestions** - Real-time correction recommendations

### 3. Multi-Column Validation

- **Validate multiple columns** - Different validators per column
- **Auto column mapping** - Detects column-to-validator matches
- **Cell-level tracking** - Not just row-level validation
- **Interactive editing** - Click any cell to modify
- **Visual feedback** - Color-coded results (green/red/orange)
- **Per-column summaries** - Quality bars and stats per column

### 4. Simple & Clean Architecture

- **4 core files** instead of 22
- **~1,740 lines** of clean code
- **Easy to understand** - No complex abstractions
- **Easy to modify** - Single layer architecture

### 5. Private & Secure

- **100% local** - No API calls, no cloud services
- **Your data stays yours** - Never leaves your machine
- **No telemetry** - No usage tracking

### 6. Works for ANY Data Type

Train validators for:

- Phone numbers (any country format)
- Email addresses
- Person names
- Addresses (street, city, state, ZIP)
- Ages and numeric ranges
- Medical data (blood sugar, vitals)
- Company/product names
- Dates, codes, IDs
- Custom business formats

## Usage Examples

### Example 1: Train a Phone Validator

```python
from ml import GenericMLValidator

validator = GenericMLValidator()

training_data = [
    ("+6591234567", "valid"),
    ("+65 9123 4567", "valid"),
    ("123", "invalid"),
    ("abc", "invalid"),
]

validator.train(training_data, "phone")
validator.save("models/phone_validator.pkl")

is_valid, confidence = validator.validate("+6598765432")
print(f"Valid: {is_valid}, Confidence: {confidence:.1%}")
```

### Example 2: Train from CSV

```python
from ml import GenericMLValidator

validator = GenericMLValidator()

validator.train_from_csv(
    "training_data/email_training.csv",
    text_col="text",
    label_col="label",
    data_type="email"
)

validator.save("models/email_validator.pkl")
```

### Example 3: Batch Validation

```python
from ml import GenericMLValidator

validator = GenericMLValidator("models/phone_validator.pkl")

phone_numbers = ["+6591234567", "123", "+65 9876 5432", "invalid"]
results = validator.validate_batch(phone_numbers)

for phone, (is_valid, confidence) in zip(phone_numbers, results):
    status = "VALID" if is_valid else "INVALID"
    print(f"{phone} -> {status} ({confidence:.1%})")
```

### Example 4: Corrections

```python
from ml import GenericMLCorrector

corrector = GenericMLCorrector()

corrected = corrector.correct("1234567e90")  # e -> 3
print(f"Corrected: {corrected}")  # "1234567390"
```

## Tech Stack

- **Python 3.8+** - Runtime
- **Streamlit 1.28.0+** - Web UI framework
- **scikit-learn 1.3.0+** - Logistic Regression classifier
- **pandas 2.0.0+** - Data manipulation
- **numpy 1.24.0+** - Numerical computing
- **streamlit-aggrid 0.3.4+** - Interactive editable data grid
- **joblib 1.3.0+** - Model serialization
- **openpyxl 3.1.0+** - Excel file support

**Total**: 7 core packages

## Comparison: Before vs After Revamp

| Aspect               | Before (Complex)                                  | After (Simple)                                |
| -------------------- | ------------------------------------------------- | --------------------------------------------- |
| **Files**            | 22+ files across multiple layers                  | 4 core files + 1 app                          |
| **Lines of Code**    | ~3,000+ lines                                     | ~1,740 lines                                  |
| **Architecture**     | 3 layers (core → plugins → base_validators)       | Single layer (ml/)                            |
| **Dependencies**     | Many heavy libraries (spaCy, transformers, torch)  | Lightweight (scikit-learn, pandas, streamlit) |
| **Features**         | 51 features                                       | 71 features                                   |
| **Approach**         | Pre-trained NLP models                            | Train on YOUR data                            |
| **Validation**       | Row-level only                                    | Cell-level + whitelist hybrid                 |
| **UI**               | Basic display                                     | Interactive grid with corrections             |

## Design Philosophy

### "Train on YOUR data, validate forever"

**Core principles:**

1. **Simplicity over complexity** - Fewer files, cleaner code
2. **Transparency over magic** - Explicit is better than implicit
3. **Flexibility over assumptions** - Train on your data, your way
4. **Local over cloud** - Your data never leaves your machine
5. **Generic over specialized** - One pipeline for all data types

## FAQ

**Q: Do I need to provide training data?**
A: We include 9 pre-trained models with ~2,047 training examples. You can use these immediately or train your own.

**Q: How much training data do I need?**
A: Minimum 50-100 examples (mix of valid and invalid). More is better. Our validators range from 70-565 examples.

**Q: Can I validate multiple data types in one CSV?**
A: Yes! Map different columns to different validators. For example: phone column → phone validator, email column → email validator.

**Q: How does the hybrid validation work?**
A: Known valid values are matched exactly via a whitelist (100% confidence). Unknown values go through the ML model for classification.
