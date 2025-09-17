# ML-Data-Validator

A clean, professional Streamlit-based data validation tool that uses Machine Learning to detect and suggest corrections for invalid survey data. Built specifically for cleaning CSV/Excel files before feeding them into analytics pipelines.

## Features

- **Smart Data Upload** - Support for CSV and Excel files
- **ML-Powered Validation** - Logistic Regression model for intelligent phone number validation
- **Interactive Data Editor** - Click-to-edit table with real-time validation
- **One-Click Suggestions** - AI-generated corrections with instant apply buttons
- **Dual Editing Modes** - Auto-apply suggestions OR manual table editing
- **Visual Data Quality Dashboard** - Clear metrics and confidence scores
- **Professional Export** - Clean data export with validation reports
- **Robust Error Handling** - Graceful handling of mixed data types and edge cases

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
- **Phone Numbers**: Machine Learning (Logistic Regression) with rule-based fallback
- **Blood Sugar**: Rule-based validation (50-500 mg/dL range)
- **Generic Data**: Basic validation for empty/null values

### ML Features
- **100% Training Accuracy** on phone pattern recognition
- **Confidence Scores** for each validation
- **Intelligent Suggestions** with character correction (e.g., 'e' → '3')
- **Real-time Re-validation** after manual edits

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
- **Features**: 10 engineered features (length, digit count, format patterns, etc.)
- **Training Data**: 3000 synthetic phone number examples with + sign and at least 7 digits rule
- **Performance**: 100% accuracy on validation set

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
| Phone Numbers | ML + Rules | `+1234567890` | `+1555444ee3` |
| Blood Sugar | Rule-based | `85` | `ewqewq` |
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
- **Expandable Sections** - Organized by data column with issue counts
- **One-Click Apply** - Individual "Apply" buttons for each suggestion
- **Bulk Operations** - "Apply All" buttons for multiple suggestions
- **Instant Updates** - Real-time table refresh after applying suggestions
- **Smart Filtering** - Only shows actionable, high-quality suggestions

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
# Input CSV with issues:
PhoneNumber,BloodSugar
+1234567890,95        # Valid
+1555444ee3,ewqewq    # Invalid
1234567890,-10        # Invalid

# ML Suggestions Display:
Row 6    Current: +1555444ee3    Suggested: +1555444333    [Apply]
Row 7    Current: ewqewq         Suggested: 85             [Apply]
Row 8    Current: -10            Suggested: 80             [Apply]

# After clicking Apply buttons:
PhoneNumber,BloodSugar
+1234567890,95        # Unchanged (valid)
+1555444333,85        # Fixed with one-click apply
+11234567890,80       # Fixed with one-click apply
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

### Recent Improvements
- **One-Click Apply Functionality** - Added instant suggestion application
- **Complete code cleanup** - Removed debugging clutter and complex logic
- **Fixed string concatenation errors** - Eliminated f-string type mixing issues
- **Simplified architecture** - Focus on core functionality
- **Enhanced error handling** - Graceful failure recovery

### Architecture Principles
- **Separation of Concerns** - Training vs. validation logic separated
- **Defensive Programming** - Extensive error handling and type safety
- **User-Centric Design** - Simple, intuitive interface
- **Maintainable Code** - Clean, readable implementation

## Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
scikit-learn>=1.3.0
streamlit-aggrid>=0.3.4
phonenumbers>=8.13.0
openpyxl>=3.1.0
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