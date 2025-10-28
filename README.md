# ML-Based CSV Data Validator

A professional Streamlit application that uses **ML-powered validation and correction** to clean data in CSV/Excel files. Features intelligent phone number correction, numeric range validation, email checking, and more through a plugin-based architecture.

---

## Quick Start

### First-Time Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd ML-data-validator

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch application
streamlit run app.py
```

### Returning Users

```bash
# 1. Navigate to project directory
cd ML-data-validator

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Launch application
streamlit run app.py
```

**Note:** Always activate the virtual environment before running the app. You'll see `(venv)` in your terminal prompt when activated.

Access the app at `http://localhost:8501`

---

## Troubleshooting

### ModuleNotFoundError after git pull

```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install/update dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Virtual environment won't activate (Windows PowerShell)

```bash
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
venv\Scripts\activate
```

---

## Features

- **ML-Powered Phone Correction** - XGBoost character-level correction
- **Numeric Range Validation** - Age, height, weight, blood sugar, calories, heart rate, steps, temperature, blood pressure
- **Email Validation** - RFC-compliant pattern matching
- **Date Validation** - Multi-format date parsing
- **Auto Column Type Detection** - Intelligent data type detection from column names and content
- **Real-Time Interactive Editor** - Click-to-edit with color-coded validation
- **ML Suggestions** - Smart corrections with confidence scores
- **Export Clean Data** - CSV/Excel export

---

## Project Structure

```
ML-data-validator/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── ml/                             # ML Backend
│   ├── validators/                 # Plugin system
│   │   ├── __init__.py
│   │   ├── phone_validator_plugin.py
│   │   ├── phone_corrector_plugin.py
│   │   ├── email_validator.py
│   │   ├── date_validator.py
│   │   ├── numeric_range_validator.py
│   │   └── numeric_range_corrector.py
│   │
│   ├── base_validator.py           # Base validator interface
│   ├── base_corrector.py           # Base corrector interface
│   ├── validator_registry.py       # Plugin registry (singleton)
│   ├── init_validators.py          # Initialize & register plugins
│   ├── column_type_detector.py     # Auto column type detection
│   ├── validator.py                # Phone validator (Logistic Regression)
│   ├── edit_distance_corrector.py  # Phone corrector (XGBoost)
│   ├── feature_extractor.py        # Feature engineering
│   ├── model_trainer.py            # Training utilities
│   └── saved_models/               # ML model configs
│       └── phone_suggestion_model_config.pkl
│
├── saved_models/                   # Trained models
│   └── phone_validator_model.pkl   # Logistic Regression model
│
├── data/                           # Training & test data
│   ├── raw/                        # Test datasets
│   │   ├── phone_training_data_improved.csv
│   │   ├── test_data_multitype.csv
│   │   ├── test_different_order.csv
│   │   └── test_mixed_columns.csv
│   ├── logistic_regression_training.csv
│   ├── xgboost_corrector_training.csv
│   └── test_data.csv
│
└── scripts/                        # Utility scripts
    └── generate_aligned_training_data.py
```

---

## Supported Data Types

### Phone Numbers
- Validator: Logistic Regression ML model
- Corrector: XGBoost character-level edit distance
- Validates international format with country codes

### Email Addresses
- Validator: RFC-compliant pattern matching
- No automatic correction (manual edit required)

### Numeric Ranges
- **Age**: 0-120 years
- **Height**: 140-210 cm
- **Weight**: 30-200 kg
- **Blood Sugar**: 70-180 mg/dL
- **Calories**: 1000-4000 kcal
- **Heart Rate**: 60-100 bpm
- **Steps**: 1000-20000 per day
- **Temperature**: 36-38°C
- **Blood Pressure**: Systolic/Diastolic

### Dates
- Multi-format parser (YYYY-MM-DD, DD/MM/YYYY, etc.)
- Automatic format detection

---

## License

This project is designed for data cleaning and validation tasks.
