# ML-Based CSV Data Validator

A professional Streamlit application that uses **ML-powered validation and correction** to clean data in CSV/Excel files. Features intelligent phone number correction, numeric range validation, email checking, and more through a plugin-based architecture.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Using the Application](#using-the-application)
- [Model Training](#model-training)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [System Requirements](#system-requirements)

---

## Features

- **Plugin-Based Architecture** - Extensible validator and corrector system
- **ML-Powered Phone Correction** - Logistic Regression validator + XGBoost corrector
- **Numeric Range Validation** - Automatic detection of age, height, weight, blood sugar, calories, etc.
- **Email Validation** - RFC-compliant pattern matching
- **Date Support** - Multi-format date parsing
- **Auto Column Type Detection** - Intelligently detects data types from column names and content
- **Real-Time Interactive Editor** - Click-to-edit table with color-coded validation
- **ML-Powered Suggestions** - Intelligent corrections with confidence scores
- **Visual Quality Dashboard** - Overall quality metrics and column-wise accuracy
- **Professional Export** - Clean data export to CSV/Excel

---

## Quick Start

### Installation (5 minutes)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch application
streamlit run app.py
```

Access at `http://localhost:8501`

---

## Installation

### Prerequisites

- **Python**: 3.8 or higher (tested on 3.10)
- **pip**: Latest version recommended
- **RAM**: 4 GB minimum, 8 GB recommended
- **Disk Space**: ~1.5 GB (for all dependencies)

### Step-by-Step Installation

#### 1. Python Version Check

```bash
python --version
# Should show Python 3.8 or higher
```

If not installed, download from [python.org](https://www.python.org/downloads/)

#### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Verify activation (prompt should show (venv))
```

#### 3. Install Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# This will take 5-10 minutes
```

#### 4. Verify Installation

```bash
# Check if all packages are installed
pip list

# Check if models exist
ls saved_models/
# Should show:
# - phone_validator_model.pkl
# - edit_distance_corrector.pkl
```

#### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at: `http://localhost:8501`

### Common Installation Issues

#### Problem: Permission denied on Windows
**Solution**: Run Command Prompt as Administrator

#### Problem: C++ build tools required (python-Levenshtein)
**Windows**:
- Download "Microsoft C++ Build Tools" from Microsoft
- Or use pre-built wheels: `pip install python-Levenshtein --only-binary :all:`

**macOS**:
```bash
xcode-select --install
```

**Linux**:
```bash
sudo apt-get install build-essential  # Debian/Ubuntu
sudo yum install gcc gcc-c++          # RedHat/CentOS
```

---

## How It Works

### Plugin-Based Validation Pipeline

```
Upload CSV/Excel
    |
    v
Column Type Detection
    |
    +-- Analyzes column names and data patterns
    +-- Detects: phone, email, age, height, weight, blood sugar, etc.
    |
    v
Validation (Stage 1)
    |
    +-- Phone Validator (Logistic Regression ML)
    +-- Email Validator (RFC pattern matching)
    +-- Numeric Validator (Range checking)
    +-- Date Validator (Multi-format parsing)
    |
    v
Invalid Data Identified
    |
    v
Correction Suggestions (Stage 2)
    |
    +-- Phone Corrector (XGBoost character-level)
    +-- Numeric Corrector (Statistical adjustment)
    +-- No corrector available → Manual input required
    |
    v
Apply Suggestions or Edit Manually
    |
    v
Export Clean Data
```

### Supported Data Types

**Phone Numbers:**
- International format validation (+1234567890)
- ML-based correction for typos and missing digits
- XGBoost character-level edit distance model

**Email Addresses:**
- RFC-compliant pattern matching
- Domain validation
- No automatic correction (manual edit required)

**Numeric Ranges:**
- **Age**: 0-120 years
- **Height**: 140-210 cm
- **Weight**: 30-200 kg
- **Blood Sugar**: 70-180 mg/dL
- **Temperature**: 36-38°C
- **Heart Rate**: 60-100 bpm
- **Calories**: 1000-4000 kcal
- **Steps**: 1000-20000 per day

**Dates:**
- Multi-format parser (YYYY-MM-DD, DD/MM/YYYY, etc.)
- Automatic format detection

---

## Using the Application

### 1. Launch Application

```bash
streamlit run app.py
```

### 2. Upload Data

- Click "Upload your CSV or Excel file"
- System automatically detects column types
- Shows Data Quality Dashboard with overall metrics

### 3. View Validation Results

**Data Quality Dashboard:**
- Overall quality percentage
- Valid/invalid cell counts
- Column-wise accuracy metrics

**Interactive Data Table:**
- **Red cells**: Invalid data
- **Orange cells**: Modified data (after applying suggestions or manual edits)
- **Green cells**: Valid data
- Click any cell to edit manually
- Auto-revalidation on changes

### 4. Review ML Suggestions

**Suggestion Display:**
- Grouped by column
- Shows: Row | Current Value | Suggested Value | Confidence | [Apply]
- Items without suggestions show "Need Manual Input"
- **Apply All** buttons at top of each column
- **Global Apply All** button applies all suggestions across all columns

**Example:**
```
Row 5  | Current: 1234567890   | Suggested: +11234567890 | 95% | [Apply]
Row 8  | Current: henry@invalid | Need Manual Input       |     |
```

### 5. Export Clean Data

- **Download Clean CSV**: Data without validation columns
- **Download Excel**: Clean data in Excel format
- Timestamp automatically added to filename

---

## Model Training

The application includes an integrated training interface in the **Model Training** tab.

### Training Phone Validator (Logistic Regression)

1. Go to "Model Training" tab
2. Select "Phone Validator (Logistic Regression)"
3. Upload training CSV with format:
   ```csv
   phone,is_valid
   +1234567890,1
   +123,0
   abc123,0
   ```
4. Click "Train Logistic Regression Model"
5. Model automatically saves to `saved_models/phone_validator_model.pkl`

### Training Phone Corrector (XGBoost)

1. Go to "Model Training" tab
2. Select "Phone Corrector (XGBoost)"
3. Upload training CSV with format:
   ```csv
   invalid,valid,operations
   1234567890,+11234567890,insert_country|keep|keep...
   ```
4. Click "Train XGBoost Model"
5. Model automatically saves to `saved_models/edit_distance_corrector.pkl`

### Training Data Locations

- `data/logistic_regression_training.csv` - Phone validator training data
- `data/xgboost_corrector_training.csv` - Phone corrector training data
- `data/raw/` - Test datasets

---

## Project Structure

```
ml-data-validator/
├── app.py                          # Main Streamlit application
├── requirements.txt                # All dependencies
├── README.md                       # This file
│
├── ml/                             # ML modules
│   ├── validators/                 # Plugin-based validators & correctors
│   │   ├── phone_validator_plugin.py
│   │   ├── phone_corrector_plugin.py
│   │   ├── email_validator.py
│   │   ├── date_validator.py
│   │   ├── numeric_range_validator.py
│   │   └── numeric_range_corrector.py
│   │
│   ├── base_validator.py           # Base validator interface
│   ├── base_corrector.py           # Base corrector interface
│   ├── validator_registry.py       # Plugin registry system
│   ├── init_validators.py          # Initialization
│   ├── column_type_detector.py     # Auto column type detection
│   ├── validator.py                # Phone validator (Logistic Regression)
│   ├── edit_distance_corrector.py  # Phone corrector (XGBoost)
│   ├── feature_extractor.py        # Feature engineering
│   ├── model_trainer.py            # Training utilities
│   └── generate_aligned_training_data.py
│
├── saved_models/                   # Trained models
│   ├── phone_validator_model.pkl   # Logistic Regression model
│   └── edit_distance_corrector.pkl # XGBoost model
│
├── data/                           # Training & test data
│   ├── raw/                        # Test datasets
│   │   ├── test_data_multitype.csv
│   │   ├── test_different_order.csv
│   │   └── test_mixed_columns.csv
│   ├── logistic_regression_training.csv
│   ├── xgboost_corrector_training.csv
│   └── test_data.csv
│
└── utils/                          # Utility functions
    └── phone_corrector.py
```

---

## Troubleshooting

### Issue: Apply Button Doesn't Update Grid

**Symptoms**: Clicking "Apply" doesn't visually update the table

**Solution**: This has been fixed in the latest version with:
- Versioned grid keys that force refresh
- Automatic rerun after applying suggestions
- Orange highlighting for modified cells

**To verify the fix:**
1. Upload test data
2. Click "Apply" on any suggestion
3. Grid should refresh showing orange cells for modified data

**If still not working:**
```bash
# Clear browser cache (Ctrl+Shift+R)
# Or restart Streamlit
streamlit run app.py
```

### Issue: No Suggestions Appear

**Possible causes:**

1. **No corrector available for that data type**
   - Check System Status tab to see which correctors are available
   - Only phone numbers have ML correctors currently
   - Email and other types require manual editing

2. **Value too corrupted to correct**
   - Phone: "abc123xyz" → Cannot fix
   - Numeric: "-999999" → Outside reasonable range

3. **Model not loaded**
   - Check System Status tab for model loading status
   - Train models using the Model Training tab

### Issue: Wrong Column Type Detected

**Solutions:**

1. **Use descriptive column names**
   ```csv
   # Instead of:
   Data,Value,Amount

   # Use:
   BloodSugar,Height,Calories
   ```

2. **Ensure 70%+ valid examples** in each column

3. **Clean mixed data** before uploading

### Issue: Large Files Slow to Process

**Performance benchmarks:**
- Small (< 1,000 rows): < 1 second
- Medium (1,000-10,000 rows): 1-5 seconds
- Large (10,000-100,000 rows): 5-30 seconds

**Optimization tips:**
- Process in chunks for very large files
- Sample first 10,000 rows for initial validation

### Issue: Grid Won't Update After Manual Edits

**Solution:**
1. Make the edit
2. Click outside the cell
3. System auto-revalidates
4. Or apply any suggestion to force refresh

### Issue: Model Not Found Errors

```
[FAIL] Phone Validator: Not loaded (train model first)
```

**Solution:**
1. Go to "Model Training" tab
2. Use existing training data or upload your own
3. Train the required model
4. Model automatically saves and reloads

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.8 | 3.10+ |
| **RAM** | 4 GB | 8 GB+ |
| **Disk Space** | 2 GB | 5 GB+ |
| **CPU** | Dual-core | Quad-core+ |
| **Internet** | Required for install | Not required after |

### Package Sizes

| Package | Size | Purpose |
|---------|------|---------|
| numpy | ~20 MB | Numerical computing |
| pandas | ~50 MB | Data manipulation |
| scikit-learn | ~30 MB | ML algorithms |
| xgboost | ~100 MB | Gradient boosting |
| streamlit | ~50 MB | Web UI |
| streamlit-aggrid | ~10 MB | Interactive grid |

**Total Installation**: ~1.5 GB

---

## Performance

- **Model Loading**: ~1-2 seconds (all models)
- **Column Type Detection**: < 100ms
- **Validation**: ~50-100ms per 1000 rows
- **Suggestion Generation**: ~100-200ms per invalid cell
- **Total Pipeline**: < 1 second for typical files (100-1000 rows)

### Memory Usage

- Base app: ~200 MB
- With models loaded: ~400 MB
- Large file (10K rows): ~600 MB

---

## Technologies

**Backend:**
- Python 3.8+
- NumPy, Pandas, SciPy

**ML Frameworks:**
- Scikit-learn (Logistic Regression, preprocessing)
- XGBoost (character-level correction)

**Frontend:**
- Streamlit (web UI framework)
- streamlit-aggrid (interactive data grid)

**Data Processing:**
- openpyxl (Excel support)
- fuzzywuzzy (fuzzy matching)
- python-Levenshtein (edit distance)
- phonenumbers (phone validation)

---

## Architecture Principles

**Plugin-Based System:**
- Easy to add new validators and correctors
- Clean separation of concerns
- Validators and correctors registered dynamically

**Validation Pipeline:**
- Clear separation between validation and correction
- Validation runs on all data
- Suggestions only for invalid data

**Fallback Mechanisms:**
- Rule-based validation if models not loaded
- Edit distance fallback for text correction
- Graceful degradation

---

## Debug Mode

To enable debug output, add temporarily to app.py:

```python
st.write("Debug Info:")
st.write("Session state keys:", list(st.session_state.keys()))
st.write("Data version:", st.session_state.get('data_version', 0))
st.write("Registry status:", st.session_state.registry)
```

---

## Cleanup / Uninstall

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
# Windows:
rmdir /s venv
# macOS/Linux:
rm -rf venv
```

---

## Future Enhancements

- [ ] Add more data types (URL, SSN, ZIP code)
- [ ] Support for custom validation rules
- [ ] Multi-language support
- [ ] Active learning for model improvement
- [ ] Batch processing API
- [ ] REST API endpoints

---

## License

This project is designed for data cleaning and validation tasks.

---

## Getting Help

If you encounter issues:

1. Check this README's Troubleshooting section
2. Verify Python version compatibility (3.8+)
3. Check System Status tab for model loading status
4. Try test data files in `data/raw/`
5. Check console/terminal for error messages

---

**Ready to validate your data?**

```bash
streamlit run app.py
```

Upload your CSV/Excel file and let the ML pipeline clean your data!
