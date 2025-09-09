# ML-Data-Validator

A clean, professional Streamlit-based data validation tool that uses Machine Learning to detect and suggest corrections for invalid survey data. Built specifically for cleaning CSV/Excel files before feeding them into analytics pipelines.

## 🎯 Features

- **Smart Data Upload** - Support for CSV and Excel files
- **ML-Powered Validation** - Random Forest model for intelligent phone number validation
- **Interactive Data Editor** - Click-to-edit table with real-time validation
- **Intelligent Suggestions** - AI-generated correction suggestions for invalid data
- **Visual Data Quality Dashboard** - Clear metrics and confidence scores
- **Professional Export** - Clean data export with validation reports
- **Robust Error Handling** - Graceful handling of mixed data types and edge cases

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Train the ML Model (One-time setup)
```bash
cd ml
python train_model.py
```
This creates the Random Forest model file: `ml/trained_models/phone_validator_model.pkl`

### 2. Run the Application
```bash
streamlit run streamlit_app_with_ml.py
```

### 3. Upload and Validate Data
1. Upload your CSV/Excel file
2. View validation results with red highlighting for invalid data
3. Check ML-powered suggestions below the table
4. Edit data directly in the table
5. Export clean, validated data

## 📊 How It Works

### Workflow
```
Upload CSV → ML Validation → Interactive Dashboard → Manual Editing → Clean Export
```

### Validation Methods
- **Phone Numbers**: Machine Learning (Random Forest) + Rule-based fallback
- **Blood Sugar**: Rule-based validation (50-500 mg/dL range)
- **Generic Data**: Basic validation for empty/null values

### ML Features
- **100% Training Accuracy** on phone pattern recognition
- **Confidence Scores** for each validation
- **Intelligent Suggestions** with character correction (e.g., 'e' → '3')
- **Real-time Re-validation** after manual edits

## 🏗️ Project Structure

```
ML-Data-Validator/
├── streamlit_app_with_ml.py     # Main Streamlit application (clean version)
├── streamlit_app_messy_backup.py # Previous messy version (backup)
├── requirements.txt              # Python dependencies
├── README.md                    # This documentation
├── 
├── ml/                          # Machine Learning components
│   ├── phone_validator.py       # ML phone validation (inference only)
│   ├── phone_model_trainer.py   # ML model training
│   ├── train_model.py          # High-level training interface
│   └── trained_models/         # Stored ML models
│       └── phone_validator_model.pkl
├── 
├── utils/                       # Validation utilities
│   ├── phone_validator.py      # Rule-based phone validation
│   └── blood_sugar_validator.py # Rule-based blood sugar validation
├── 
└── data/                        # Sample and test data
    └── sample_data/
        └── synthetic_testing_validation_phone.csv
```

## 🔧 Technical Implementation

### Core Technologies
- **Streamlit** - Web interface
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine Learning (Random Forest)
- **AgGrid** - Interactive data table
- **Phonenumbers** - Phone validation library

### ML Model Details
- **Algorithm**: Random Forest Classifier
- **Features**: 10 engineered features (length, digit count, format patterns, etc.)
- **Training Data**: 2000 synthetic phone number examples
- **Performance**: 100% accuracy on validation set

### Smart Phone Suggestions
```python
# Example corrections:
"+1555444ee3" → "+1555444333"  # Character substitution (e→3)
"1234567890"  → "+11234567890" # Add country code
"+65 invalid" → "+6512345678"  # Pattern completion
```

## 📱 Supported Data Types

| Data Type | Validation Method | Example Valid | Example Invalid |
|-----------|------------------|---------------|-----------------|
| Phone Numbers | ML + Rules | `+1234567890` | `+1555444ee3` |
| Blood Sugar | Rule-based | `85` | `ewqewq` |
| Generic Data | Basic checks | `Valid Text` | `null` |

## 🎨 User Interface

### Dashboard Metrics
- **Overall Quality** - Percentage of valid data
- **Column-specific Accuracy** - Individual validation rates
- **ML Confidence Scores** - Model confidence for predictions

### Data Table Features
- **Color-coded Validation**:
  - 🔴 Red: Invalid data
  - 🟢 Green: High-confidence valid data (≥90%)
  - 🟠 Orange: Medium-confidence valid data (70-90%)
  - 🔵 Blue: Low-confidence valid data (<70%)
- **Click-to-Edit** - Direct cell editing with auto-revalidation
- **Session Persistence** - Edits are saved during the session

### ML Suggestions
- **Expandable Sections** - Organized by data column
- **Copy-Paste Friendly** - Easy to apply suggested corrections
- **Smart Filtering** - Only shows actionable suggestions

## 📤 Export Options

### CSV Export
- Clean data without validation columns
- Timestamped filename
- Ready for analytics pipelines

### Excel Export
- **Clean_Data** sheet - Validated data
- **ML_Validation_Report** sheet (if ML model used) - Detailed metrics

## 🔍 Example Usage

### Sample Data Processing
```python
# Input CSV with issues:
PhoneNumber,BloodSugar
+1234567890,95        # ✅ Valid
+1555444ee3,ewqewq    # ❌ Invalid
1234567890,-10        # ❌ Invalid

# After ML processing:
PhoneNumber,BloodSugar
+1234567890,95        # ✅ Unchanged (valid)
+1555444333,85        # ✅ Corrected by ML suggestions
+11234567890,80       # ✅ Fixed manually or by suggestions
```

## 🛡️ Robust Error Handling

The application handles various data challenges:
- **Mixed data types** (strings, integers, floats)
- **Missing values** (NaN, None, empty strings)
- **Invalid pandas indices** after filtering
- **Unicode and encoding issues**
- **Large file processing** with memory optimization

## 🚀 Performance Features

- **Session State Management** - Preserves data across interactions
- **Efficient Re-validation** - Only re-validates changed data
- **Batch ML Processing** - Optimized for multiple records
- **Memory-safe Export** - Handles large datasets

## 🔧 Development Notes

### Recent Improvements
- **Complete code cleanup** - Removed debugging clutter and complex logic
- **Fixed string concatenation errors** - Eliminated f-string type mixing issues
- **Simplified architecture** - Focus on core functionality
- **Enhanced error handling** - Graceful failure recovery

### Architecture Principles
- **Separation of Concerns** - Training vs. validation logic separated
- **Defensive Programming** - Extensive error handling and type safety
- **User-Centric Design** - Simple, intuitive interface
- **Maintainable Code** - Clean, readable implementation

## 📋 Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
scikit-learn>=1.3.0
streamlit-aggrid>=0.3.4
phonenumbers>=8.13.0
openpyxl>=3.1.0
```

## 🤝 Contributing

This project prioritizes clean, maintainable code. When contributing:
1. Keep functions focused and single-purpose
2. Use comprehensive error handling
3. Maintain type safety throughout
4. Test with various data formats and edge cases

## 📄 License

This project is designed for data cleaning and validation tasks in survey data processing pipelines.

---

**Ready to clean your data?** Run `streamlit run streamlit_app_with_ml.py` and upload your CSV file to get started! 🚀