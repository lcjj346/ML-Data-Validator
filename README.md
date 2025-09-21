# ML-Data-Validator

A focused, professional Streamlit application that uses **Logistic Regression Machine Learning** to validate and correct phone numbers in CSV/Excel files. Streamlined for efficiency with one-click suggestion application and real-time validation.

## Features

- **Smart Data Upload** - Support for CSV and Excel files
- **ML-Powered Phone Validation** - Logistic Regression model with 100% training accuracy
- **One-Click Apply System** - Instant suggestion application with individual and bulk "Apply" buttons
- **Real-Time Interactive Editor** - Click-to-edit table with automatic re-validation
- **Smart Suggestion Engine** - Character-level corrections (e.g., 'e' → '3') with confidence scoring
- **Visual Quality Dashboard** - Color-coded validation with ML confidence metrics
- **Streamlined Architecture** - Clean separation of training vs validation logic
- **Professional Export** - Clean data export ready for analytics pipelines

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Train the ML Model (One-time setup)
```bash
cd ml
python model_trainer.py
```
This creates the Logistic Regression model file: `saved_models/phone_validator_model.pkl`

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Upload and Validate Data
1. Upload your CSV/Excel file
2. View validation results with red highlighting for invalid data
3. Check ML-powered suggestions with one-click apply buttons
4. Either apply suggestions automatically or edit data manually in the table
5. Export clean, validated data

## How It Works

### Workflow
```
Upload CSV → ML Validation → Interactive Dashboard → Auto-Apply Suggestions OR Manual Editing → Clean Export
```

### Validation Methods
- **Phone Numbers**: Machine Learning (Logistic Regression) with feature engineering
- **Generic Data**: Basic validation for empty/null values

### ML Features
- **100% Training Accuracy** on synthetic phone validation dataset
- **10 Engineered Features** - Length, digit patterns, format validation, etc.
- **Confidence Scoring** - Each prediction includes ML confidence percentage
- **Smart Character Corrections** - Intelligent substitutions (e.g., 'e' → '3', 'o' → '0')
- **Automatic Re-validation** - Instant validation after any data changes

## Project Structure

```
ML-Data-Validator/
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation
├──
├── ml/                          # Machine Learning components
│   ├── validator.py             # ML phone validation (inference only)
│   └── model_trainer.py         # ML model training
├──
├── saved_models/                # Stored ML models
│   └── phone_validator_model.pkl
├──
└── data/                        # Training and test data
    ├── training_data.csv        # Training data for ML model
    └── test_data.csv            # Test data for validation
```

## Technical Implementation

### Core Technologies
- **Streamlit** - Web interface
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine Learning (Logistic Regression)
- **AgGrid** - Interactive data table
- **Phonenumbers** - Phone validation library

### ML Model Details
- **Algorithm**: Logistic Regression Classifier
- **Features**: 10 engineered features (length, digit count, format patterns, character analysis)
- **Training Data**: 2000+ synthetic phone examples with validation rule: "+ sign + at least 7 digits"
- **Performance**: 100% accuracy on validation set with comprehensive feature engineering

### Smart Phone Suggestions
```python
# Example corrections:
"+1555444ee3" → "+1555444333"  # Character substitution (e→3)
"1234567890"  → "+11234567890" # Add country code
"+65 invalid" → "+6512345678"  # Pattern completion
```

## Supported Data Types

| Data Type | Validation Method | Example Valid | Example Invalid |
|-----------|------------------|---------------|-----------------|
| Phone Numbers | ML (Logistic Regression) | `+1234567890` | `+1555444ee3` |
| Generic Data | Basic checks | `Valid Text` | `null` |

## User Interface

### Dashboard Metrics
- **Overall Quality** - Percentage of valid data
- **Column-specific Accuracy** - Individual validation rates
- **ML Confidence Scores** - Model confidence for predictions

### Data Table Features
- **Color-coded Validation**:
  - Red: Invalid data
  - Green: High-confidence valid data (≥90%)
  - Orange: Medium-confidence valid data (70-90%)
  - Blue: Low-confidence valid data (<70%)
- **Click-to-Edit** - Direct cell editing with auto-revalidation
- **Session Persistence** - Edits are saved during the session

### ML Suggestions
- **Expandable Sections** - Organized by column with precise issue counts
- **Individual Apply Buttons** - One-click application for each suggestion
- **Bulk Apply Operations** - "Apply All" buttons for mass corrections
- **Real-Time Updates** - Instant table refresh with automatic re-validation
- **High-Quality Filtering** - Only displays actionable ML-generated suggestions

## Export Options

### CSV Export
- Clean data without validation columns
- Timestamped filename
- Ready for analytics pipelines

### Excel Export
- **Clean_Data** sheet - Validated data
- **ML_Validation_Report** sheet (if ML model used) - Detailed metrics

## Example Usage

### Sample Data Processing
```python
# Input CSV with phone number issues:
PhoneNumber
+1234567890          # Valid
+1555444ee3          # Invalid (letters in number)
1234567890           # Invalid (missing country code)

# ML Suggestions Display:
Row 2    Current: +1555444ee3    Suggested: +1555444333    [Apply]
Row 3    Current: 1234567890     Suggested: +11234567890   [Apply]

# After clicking Apply buttons:
PhoneNumber
+1234567890          # Unchanged (valid)
+1555444333          # Fixed with ML suggestion
+11234567890         # Fixed with country code addition
```

### Two Ways to Fix Data
1. **Auto-Apply Suggestions**: Click "Apply" buttons for instant fixes
2. **Manual Editing**: Click table cells to edit values directly
3. **Bulk Operations**: Use "Apply All" for multiple fixes at once

## Robust Error Handling

The application handles various data challenges:
- **Mixed data types** (strings, integers, floats)
- **Missing values** (NaN, None, empty strings)
- **Invalid pandas indices** after filtering
- **Unicode and encoding issues**
- **Large file processing** with memory optimization

## Performance Features

- **Session State Management** - Preserves data across interactions
- **Efficient Re-validation** - Only re-validates changed data
- **Batch ML Processing** - Optimized for multiple records
- **Instant Apply Operations** - Real-time suggestion application
- **Memory-safe Export** - Handles large datasets

## Development Notes

### Recent Improvements (Latest Updates)
- **Focused Single-Purpose Design** - Streamlined to phone validation only (removed blood sugar validation)
- **One-Click Apply System** - Individual and bulk suggestion application with real-time updates
- **Code Architecture Cleanup** - Complete separation of training (`model_trainer.py`) vs validation (`validator.py`)
- **Enhanced ML Pipeline** - Synthetic training data generation with 10 engineered features
- **Error Handling Improvements** - Eliminated string concatenation errors and type mixing issues
- **Dependency Optimization** - Removed unnecessary packages, kept core ML functionality

### Architecture Principles
- **Separation of Concerns** - Training vs. validation logic separated
- **Defensive Programming** - Extensive error handling and type safety
- **User-Centric Design** - Simple, intuitive interface
- **Maintainable Code** - Clean, readable implementation

## Requirements

```
streamlit
pandas
scikit-learn
streamlit-aggrid
phonenumbers
openpyxl
numpy
```

## Contributing

This project prioritizes clean, maintainable code. When contributing:
1. Keep functions focused and single-purpose
2. Use comprehensive error handling
3. Maintain type safety throughout
4. Test with various data formats and edge cases

## License

This project is designed for data cleaning and validation tasks in survey data processing pipelines.

---

**Ready to clean your data?** Run `streamlit run app.py` and upload your CSV file to get started!