# ML-Based Data Validator with NLP

A powerful Streamlit application that uses **Machine Learning** and **Natural Language Processing** to automatically validate and correct data in CSV/Excel files. Works for **ANY column type** - no configuration needed!

---

## Features

### Universal NLP Validation
- Works for ANY column type - Countries, cities, products, names, etc.
- No training required - Uses pre-trained spaCy + Sentence Transformers
- 100% Free & Private - Runs locally, no API costs
- Smart corrections - "Sinapore" → "Singapore" with confidence scores

### Specialized Validators
- **Phone Numbers** - ML-powered (Logistic Regression + XGBoost)
- **Email Addresses** - RFC validation + fuzzy domain matching
- **Numeric Ranges** - Age, height, weight, blood sugar, heart rate, etc.
- **Dates** - Multi-format parsing

### Interactive UI
- Real-time cell editing
- Color-coded validation (Red=invalid, Green=valid, Orange=edited)
- One-click corrections
- Export to CSV/Excel

---

## Quick Start

```bash
# 1. Navigate to project
cd ml-data-validator

# 2. Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model (for NLP validation)
python -m spacy download en_core_web_sm

# 5. Launch application
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## Usage Example

### 1. Upload Your CSV
Any CSV with any columns - the system auto-detects types!

### 2. See Validation Results
```
Country Column - 3 issues found (nlp_text)
Row 3: "Sinapore"  → Suggest: "Singapore" (94%)  [Apply]
Row 4: "Germny"    → Suggest: "Germany" (92%)    [Apply]

Email Column - 1 issue found (email)
Row 7: "user@homail.com" → Suggest: "user@hotmail.com" (95%) [Apply]

PhoneNumber Column - 0 issues found (phone)
```

### 3. Apply Corrections
- Click individual [Apply] buttons
- Or use "Apply All Suggestions" for batch fixes

### 4. Export Clean Data
Download as CSV or Excel

---

## How It Works

### Column Type Detection
```
User uploads CSV
    ↓
System analyzes each column:
  - Column name ("Country", "Email", "PhoneNumber")
  - Sample values
    ↓
Auto-assigns validator:
  - phone → PhoneValidator (ML-based)
  - email → EmailValidator (rule-based + fuzzy)
  - country/city/location → UniversalNLPValidator
  - unknown text → UniversalNLPValidator (semantic similarity)
```

### NLP Validation Techniques

| Column Type | Detection | Validation Method |
|------------|-----------|-------------------|
| **Country** | Column name pattern | Fuzzy match vs pycountry (195 countries) |
| **City/Location** | Column name pattern | spaCy NER + fuzzy match |
| **Person Name** | Column name pattern | spaCy NER (PERSON entities) |
| **Organization** | Column name pattern | spaCy NER (ORG entities) |
| **Unknown Text** | Fallback | Semantic similarity (embeddings) |
| **Phone** | Regex pattern | Logistic Regression (ML) |
| **Email** | Regex pattern | RFC validation + fuzzy domains |
| **Numeric** | Data analysis | Range-based validation |

---

## Supported Data Types

### Text Data (NLP-Based)
- **Countries** - Fuzzy matching against 195 countries
- **Cities/Locations** - spaCy entity recognition
- **Person Names** - spaCy person detection
- **Organizations** - spaCy organization detection
- **Products** - Semantic similarity
- **Any text column** - Falls back to similarity-based validation

### Phone Numbers (ML-Based)
- **Validator**: Logistic Regression
- **Corrector**: XGBoost character-level correction
- **Formats**: International (+1, +65, +44, etc.)

### Email Addresses (Rule-Based + Fuzzy)
- **Validator**: RFC-compliant regex
- **Corrector**: Fuzzy domain matching
- **Catches**: homail→hotmail, gmial→gmail, yahooo→yahoo

### Numeric Ranges (Rule-Based)
| Type | Valid Range | Unit |
|------|-------------|------|
| Age | 0-120 | years |
| Height | 140-210 | cm |
| Weight | 30-200 | kg |
| Blood Sugar | 70-180 | mg/dL |
| Calories | 1000-4000 | kcal |
| Heart Rate | 60-100 | bpm |
| Steps | 1000-20000 | per day |
| Temperature | 36-38 | °C |
| Blood Pressure | 90-140 / 60-90 | mmHg |

### Dates (Rule-Based)
- **Formats**: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, etc.
- **Auto-detection**: Tries multiple formats automatically

---

## Project Structure

```
ml-data-validator/
├── app.py                      # Main Streamlit application
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .gitignore
│
├── ml/                         # ML & NLP Backend
│   ├── models/                 # Trained ML models (.pkl files)
│   │   ├── phone_validator.pkl     # Logistic Regression phone validator
│   │   └── phone_corrector.pkl     # XGBoost phone corrector
│   │
│   ├── validators/             # All validators, correctors, and registry
│   │   ├── plugins/            # ML model wrappers (adapt legacy code)
│   │   │   ├── phone_validator_plugin.py    # Wraps core phone validator
│   │   │   ├── phone_corrector_plugin.py    # Wraps core phone corrector
│   │   │   └── email_corrector_plugin.py    # Email domain fuzzy matching
│   │   │
│   │   ├── base_validators/    # Core validation logic (rule-based & NLP)
│   │   │   ├── email_validator.py           # RFC email validation
│   │   │   ├── date_validator.py            # Multi-format date parsing
│   │   │   ├── numeric_range_validator.py   # Range-based validation
│   │   │   ├── numeric_range_corrector.py   # Numeric correction
│   │   │   ├── nlp_universal_validator.py   # NLP for ANY text column
│   │   │   └── nlp_universal_corrector.py   # NLP-based corrections
│   │   │
│   │   ├── registry/           # Central validator management
│   │   │   ├── validator_registry.py        # Singleton registry
│   │   │   └── init_validators.py           # Register all validators
│   │   │
│   │   └── __init__.py         # Package exports
│   │
│   ├── core/                   # Legacy phone validation code (used by plugins)
│   │   ├── phone_validator.py      # Original Logistic Regression validator
│   │   ├── phone_corrector.py      # Original XGBoost corrector
│   │   ├── phone_features.py       # Feature extraction for phone numbers
│   │   └── model_trainer.py        # Train phone ML models
│   │
│   ├── base_validator.py       # BaseValidator interface
│   ├── base_corrector.py       # BaseCorrector interface
│   └── column_type_detector.py # Auto-detect column types
│
├── data/
│   ├── samples/                # Sample CSVs for testing (upload these!)
│   │   ├── test_multitype.csv      # Tests ALL validators (phone, email, numeric, date)
│   │   ├── test_nlp.csv            # Tests NLP with intentional typos (countries, cities)
│   │   └── test_phone.csv          # Tests phone validation specifically
│   │
│   └── training/               # Training data for ML models (internal use only)
│       ├── logistic_regression_training.csv    # Phone validator training data
│       └── xgboost_corrector_training.csv      # Phone corrector training data
│
└── venv/                       # Virtual environment
```

### Architecture Explained

**Validators vs Correctors**
- **Validators**: Check if data is valid, return confidence scores
- **Correctors**: Suggest fixes for invalid data with confidence scores

**Plugins vs Base Validators**
- **Plugins** (ml/validators/plugins/): Wrap legacy ML models to fit the plugin interface
  - Example: PhoneValidatorPlugin wraps ml/core/phone_validator.py
- **Base Validators** (ml/validators/base_validators/): Direct implementations (no wrapping needed)
  - Example: EmailValidator directly implements validation logic

**Registry System**
- **ValidatorRegistry**: Central registry (singleton) that manages all validators/correctors
- **init_validators.py**: Registers all validators at app startup
- Allows easy addition of new data types without modifying app.py

**Data Files Explained**
- **test_multitype.csv**: Upload this to test phone, email, numeric ranges, dates all at once
- **test_nlp.csv**: Upload this to see NLP correct "Sinapore"→"Singapore", "Germny"→"Germany"
- **test_phone.csv**: Upload this to test phone validation specifically
- **training/**: Contains data to train ML models (you don't need to upload these)

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'spacy'"
```bash
# Activate venv first!
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### "spaCy model 'en_core_web_sm' not found"
```bash
python -m spacy download en_core_web_sm
```

### Virtual environment won't activate (Windows PowerShell)
```bash
# Run PowerShell as Administrator:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate:
venv\Scripts\activate
```

### NLP validation not working
1. Check spaCy: `python -m spacy validate`
2. Check transformers: `pip list | grep sentence`
3. Restart Streamlit app

---

## Dependencies

### Core ML/NLP
- **spacy** (3.7+) - Industrial NLP
- **sentence-transformers** (2.2+) - Semantic similarity
- **pycountry** (22.3+) - Country/language data
- **fuzzywuzzy** - Fuzzy string matching
- **scikit-learn** - ML algorithms
- **xgboost** - Gradient boosting

### Deep Learning
- **torch** (PyTorch) - Neural network backend
- **tensorflow** - TensorFlow backend

### UI & Data
- **streamlit** - Web UI
- **streamlit-aggrid** - Interactive grid
- **pandas** - Data manipulation
- **numpy** - Numerical computing

See `requirements.txt` for complete list.

---

## Adding Custom Validators

### 1. Create Validator Class
```python
# ml/validators/my_validator.py
from ml.base_validator import BaseValidator, ValidationResult

class MyCustomValidator(BaseValidator):
    def validate(self, value, column_name="", column_samples=None):
        # Your validation logic
        return ValidationResult(
            is_valid=True,
            confidence=0.95,
            error_type=None
        )

    def get_data_type(self):
        return "my_custom_type"
```

### 2. Register in init_validators.py
```python
from ml.validators.my_validator import MyCustomValidator

# In initialize_validators():
my_validator = MyCustomValidator()
registry.register_validator('my_custom_type', my_validator)
```

### 3. Update Column Type Detector
```python
# ml/column_type_detector.py
# Add pattern to detect your custom type
```

---

## Architecture Details

### Plugin System
- Each validator implements `BaseValidator` interface
- Each corrector implements `BaseCorrector` interface
- Validators/correctors registered in `ValidatorRegistry` (singleton)
- Allows easy addition of new data types

### NLP System
- **spaCy** - Entity recognition (locations, persons, orgs)
- **Sentence Transformers** - Semantic similarity for unknown types
- **pycountry** - Reference database for countries
- **FuzzyWuzzy** - String similarity matching

### Legacy Components
- `ml/core/` contains original phone validator implementations
- Wrapped by plugins to conform to new architecture
- Kept for compatibility with trained models

---

## Sample Data

Test the system with included samples:

- `data/samples/test_multitype.csv` - Multiple column types
- `data/samples/test_nlp.csv` - Countries, cities with typos

Upload these in the app to see NLP validation in action!

---

## License

This project is designed for data cleaning and validation tasks.

---

## Acknowledgments

- **spaCy** - Industrial-strength NLP
- **Sentence Transformers** - Semantic text embeddings
- **pycountry** - ISO country/language database
- **Streamlit** - Python web framework

---

**Made with ❤️ for data quality**
