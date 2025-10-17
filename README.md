# ML-Data-Validator

A professional Streamlit application that uses **Machine Learning** to validate and correct phone numbers in CSV/Excel files. Features dual ML models: Logistic Regression for validation and XGBoost for intelligent corrections.

## Features

- **Dual ML Models** - Logistic Regression validator + XGBoost corrector
- **Smart Phone Validation** - ML-powered validation with 95%+ accuracy
- **Intelligent Corrections** - Character-level XGBoost editing with context awareness
- **Integrated Training** - Train models directly in the UI with custom or default data
- **Real-Time Interactive Editor** - Click-to-edit table with automatic re-validation
- **One-Click Apply System** - Instant suggestion application with individual apply buttons
- **Visual Quality Dashboard** - Color-coded validation with confidence metrics
- **Professional Export** - Clean data export to CSV/Excel

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### First Time Setup

1. **Generate Training Data**
```bash
cd ml
python generate_aligned_training_data.py
```
This creates aligned training data in `data/` folder.

2. **Launch Application**
```bash
streamlit run app.py
```

3. **Train Models (in UI)**
   - Go to "Model Training" tab
   - Train Logistic Regression (uses existing data by default)
   - Train XGBoost (uses existing data by default)
   - Both models save automatically

4. **Validate Data**
   - Go to "Data Validation" tab
   - Upload your CSV/Excel file
   - View validation results and ML-powered suggestions
   - Apply corrections and export clean data

## How It Works

### Dual Model Architecture

```
Upload CSV
    |
    v
Logistic Regression Validator
    |
    v
Identifies Invalid Phones
    |
    v
XGBoost Corrector (Primary)
    |
    v
Character-Level Corrections
    |
    v
Rule-Based Fallback (if needed)
    |
    v
Export Clean Data
```

### Model 1: Logistic Regression (Validator)

**Purpose:** Classify phone numbers as valid or invalid

**Features:**
- 10 engineered features (length, digit count, format patterns)
- Binary classification (valid/invalid)
- 95-100% accuracy on training data
- Fast inference

**Training Data Format:**
```csv
phone,is_valid
+1234567890,1
invalid_phone,0
```

### Model 2: XGBoost (Corrector)

**Purpose:** Fix invalid phone numbers character-by-character

**How It Works:**
- Analyzes each character position independently
- Predicts edit operations: keep, delete, replace_0-9, insert_country
- 16 features per character (position, type, context, typo likelihood)
- Context-aware corrections (considers surrounding characters)

**Training Data Format:**
```csv
invalid,valid,operations
1234567890,+11234567890,insert_country|keep|keep|...
+1234567e90,+1234567890,keep|keep|...|replace_8|keep
```

**Example Corrections:**
```
Input:  "555-123-4567"       Output: "+15551234567"
Input:  "+1 555 123 4567"    Output: "+15551234567"
Input:  "+65 9123 4567"      Output: "+6591234567"
Input:  "1234567890"         Output: "+1234567890"
```

## Project Structure

```
ml-data-validator/
├── app.py                                  # Main Streamlit app
├── requirements.txt                        # Dependencies
├── README.md                               # This file
│
├── ml/                                     # ML components
│   ├── validator.py                        # Logistic Regression validator
│   ├── model_trainer.py                    # Logistic Regression trainer
│   ├── edit_distance_corrector.py          # XGBoost corrector
│   ├── train_ui.py                         # Training UI module
│   └── generate_aligned_training_data.py   # Data generator
│
├── saved_models/                           # Trained models
│   ├── phone_validator_model.pkl           # Logistic Regression
│   └── edit_distance_corrector.pkl         # XGBoost
│
└── data/                                   # Training data
    ├── logistic_regression_training.csv    # 10K samples
    ├── xgboost_corrector_training.csv      # 5K correction pairs
    └── test_data.csv                       # Test dataset
```

## Training Models

### Option 1: UI Training (Recommended)

1. Run `streamlit run app.py`
2. Go to "Model Training" tab
3. Select "Use existing data (Recommended)"
4. Click "Train Logistic Regression Model"
5. Click "Train XGBoost Model"
6. Refresh page to load new models

### Option 2: Command Line Training

```bash
# Generate training data
cd ml
python generate_aligned_training_data.py

# Train Logistic Regression
python model_trainer.py

# Train XGBoost
python edit_distance_corrector.py
```

### Training Data Format

**Logistic Regression:**
- Columns: `phone`, `is_valid`
- Example: `+1234567890,1` or `invalid_phone,0`

**XGBoost:**
- Columns: `invalid`, `valid`, `operations`
- Example: `1234567890,+11234567890,insert_country|keep|...`

### Custom Training Data

You can upload your own training data in the UI:
1. Go to "Model Training" tab
2. Select "Upload custom data"
3. Upload CSV file with correct format
4. Preview data and train

## Data Alignment

**IMPORTANT:** Both models are trained on aligned data, meaning:
- Every invalid phone in XGBoost training is also in Logistic training (marked as invalid)
- This ensures Logistic flags phones as invalid, and XGBoost knows how to fix them
- Prevents situations where models disagree on validity

## Technical Details

### Technologies
- **Streamlit** - Web interface
- **Pandas** - Data manipulation
- **Scikit-learn** - Logistic Regression
- **XGBoost** - Gradient boosting
- **AgGrid** - Interactive data table
- **NumPy** - Numerical operations

### ML Models

**Logistic Regression:**
- Algorithm: Binary classification
- Features: 10 engineered features
- Training: 10,000 samples (5K valid, 5K invalid)
- Accuracy: 95-100%
- Inference: < 1ms per phone

**XGBoost:**
- Algorithm: Multi-class gradient boosting
- Classes: 14 edit operations
- Features: 16 per character position
- Training: 5,000 correction pairs
- Accuracy: 85-95%
- n_estimators: 100 trees

### Correction Examples

**Missing Country Code:**
```
Input:  1234567890
Output: +1234567890
```

**Letter Typos:**
```
Input:  +1234567e9o
Output: +1234567890
```

**Extra Spaces:**
```
Input:  +1 555 123 4567
Output: +15551234567
```

**Dashes/Parentheses:**
```
Input:  (555) 123-4567
Output: +15551234567
```

## User Interface

### Data Validation Tab

**Status Display:**
- Logistic Regression (Validator): ACTIVE
- XGBoost (Corrector): ACTIVE - Used for suggestions

**Upload:**
- CSV/Excel file upload
- Automatic format detection

**Validation Results:**
- Color-coded invalid phones (red)
- Confidence scores
- Quality metrics

**Suggestions:**
- ML-powered corrections from XGBoost
- Individual "Apply" buttons
- Rule-based fallback for edge cases

**Export:**
- CSV export
- Excel export with validation report

### Model Training Tab

**Logistic Regression Training:**
- Data source selection (existing/custom)
- Data preview with statistics
- One-click training
- Accuracy display
- Auto-save and reload

**XGBoost Training:**
- Data source selection
- Data preview
- One-click training
- Accuracy display
- Auto-save and reload

## Export Options

### CSV Export
- Clean data without validation columns
- Original column names preserved
- Ready for analytics

### Excel Export
- Sheet 1: Clean_Data
- Sheet 2: ML_Validation_Report (metrics and statistics)

## Supported Phone Formats

**Valid Examples:**
- `+1234567890` - US with country code
- `+6591234567` - Singapore
- `+441234567890` - UK
- `+33123456789` - France

**Invalid Examples (will be corrected):**
- `1234567890` - Missing +
- `555-123-4567` - Has dashes
- `(555) 123-4567` - Has parentheses
- `+1 555 123 4567` - Has spaces
- `+1234567e90` - Has letters

## Error Handling

The application handles:
- Mixed data types
- Missing values (NaN, None, empty)
- Unicode characters
- Large files
- Model loading failures
- Invalid CSV formats

## Performance

- Session state management for data persistence
- Efficient re-validation on edits
- Batch ML processing
- Memory-safe export for large datasets
- Fast character-level XGBoost inference

## Requirements

```
streamlit
pandas
scikit-learn
xgboost
streamlit-aggrid
phonenumbers
openpyxl
numpy
joblib
```

## Troubleshooting

### No Suggestions Showing

**Problem:** Models mark everything as valid

**Solution:** Retrain both models with aligned data
```bash
cd ml
python generate_aligned_training_data.py
# Then train both models in the UI
```

### Model Not Loading

**Problem:** "Model not found" error

**Solution:** Train the models first
1. Generate data: `python ml/generate_aligned_training_data.py`
2. Train in UI or command line
3. Refresh the app

### Low Accuracy

**Problem:** Model accuracy < 80%

**Solution:**
- Check training data format
- Ensure 2000+ samples for Logistic Regression
- Ensure 5000+ samples for XGBoost
- Verify data alignment

## Development

### Code Cleanup (Recent)
- Removed old misaligned data generator
- Removed all emojis from UI and code
- Removed debug/test files
- Consolidated documentation
- Updated file references to use aligned data

### Architecture Principles
- Dual model approach (validation + correction)
- Aligned training data for model consistency
- UI-based training for ease of use
- Character-level correction for accuracy
- Fallback mechanisms for edge cases

## License

This project is designed for data cleaning and validation tasks.

---

**Ready to validate your data?**

```bash
streamlit run app.py
```

Upload your CSV file and let ML clean your phone numbers!
