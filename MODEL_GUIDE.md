# ML Data Validator - Model Guide

## Overview

The ML Data Validator uses a **Unified Model Architecture** where a single model file contains validation logic for all columns in your dataset. Each column gets its own:
- ML classifier (LogisticRegression)
- Feature scaler (StandardScaler)
- Whitelist of valid values
- Valid examples for correction suggestions

---

## How Validation Works

When validating a cell value, the system follows this flow:

```
1. Whitelist Check
   └─ Exact match (case-insensitive)? → VALID (100%)

2. Numeric Normalization (for numeric columns)
   └─ 95 == 95.0 == "95"? → VALID (100%)

3. Reference List Check (if column has reference list)
   └─ In reference list? → VALID (100%)

4. Typo Detection (for columns with reference lists)
   └─ >80% similar to valid value? → INVALID (shows correction)

5. ML Model Fallback
   └─ Feature extraction → Model prediction → confidence score
```

---

## Two Model Types

### 1. Base Model (`models/base_model.pkl`)
- Pre-trained on common columns: name, email, phone, country, age, address, blood_sugar
- Ships with the app, ready to use
- Developer maintains and improves this

### 2. Custom Models (`models/{name}_model.pkl`)
- Users create these for their specific datasets
- Train on any CSV structure
- Can be "stacked" with more data over time

---

## How to Improve Model Accuracy

### For Developers (Base Model)

#### 1. Add More Training Data
Edit `training_data/base_training_data.csv`:
```csv
name,email,phone,country,age,address,blood_sugar
Jean-Pierre,jp@email.fr,+33 1 2345 6789,France,45,1 Rue de Paris,95.0
```

**Tips:**
- Add diverse examples (different cultures, formats, edge cases)
- Include variations: "USA", "US", "United States"
- Add hyphenated names, compound names, single names
- Include different phone formats: +65 9123 4567, 91234567, (555) 123-4567

#### 2. Expand Reference Lists
Reference lists provide strict validation for finite value sets.

Location: `reference_lists/`

| File | Purpose | Example Values |
|------|---------|----------------|
| `country.txt` | Valid countries | Singapore, USA, UK, Germany |
| `age.txt` | Valid ages | 0-120 |
| `email_domains.txt` | Common domains | gmail.com, yahoo.com |
| `phone_prefixes.txt` | Country codes | +1, +65, +44 |

**To add values:**
```
# reference_lists/country.txt
Singapore
SG
Republic of Singapore
USA
US
United States
...
```

#### 3. Retrain the Model
After updating training data or reference lists:

```python
from ml.validator import UnifiedMLValidator
import pandas as pd

# Load updated training data
df = pd.read_csv('training_data/base_training_data.csv')

# Create and train
validator = UnifiedMLValidator()
metrics = validator.train(df, model_name='base_model')

# Save
validator.save('models/base_model.pkl')

# Check accuracy
for col, m in metrics.items():
    print(f"{col}: {m['test_accuracy']:.1%}")
```

---

### For Users (Custom Models)

#### Option 1: Create New Model
1. Go to **"Train Models"** tab
2. Upload your CSV (all rows treated as valid examples)
3. Enter a model name
4. (Optional) Exclude columns like ID, timestamp
5. Click **Train Model**

#### Option 2: Stack Data onto Existing Model
1. Go to **"Train Models"** tab
2. Select **"Add data to existing model"**
3. Upload additional CSV with more valid examples
4. Select the model to improve
5. Click **Add Training Data**

This adds new valid values to the whitelist and optionally retrains the ML model.

---

## Understanding Model Metrics

After training, you'll see per-column metrics:

| Metric | Meaning | Good Value |
|--------|---------|------------|
| Test Accuracy | % correctly classified on held-out data | >85% |
| Unique Values | Number of distinct valid values learned | Varies |

**Low accuracy causes:**
- Too few training examples
- Too little diversity in examples
- Column has unpredictable/random values

---

## Column Types & Recommendations

| Column Type | Improve With | Notes |
|-------------|--------------|-------|
| **Names** | More training data | Add cultural variations, hyphenated, single names |
| **Countries** | Reference list | Finite set - add all valid abbreviations |
| **Emails** | Training data + domain list | Patterns are learnable |
| **Phone** | Training data + prefix list | Add format variations |
| **Age** | Reference list | Finite range (0-120) |
| **Custom text** | Training data only | ML learns patterns |

---

## Typo Detection

Typo detection uses fuzzy matching (>80% similarity) but only for:
- Columns with reference lists (countries, etc.)
- NOT for open-ended columns (names, addresses)

This prevents false positives like "John Smith" being flagged as a typo of "John Smyth".

**Example:**
- "Singaproe" → 89% similar to "Singapore" → INVALID, suggests "Singapore"
- "Jean-Pierre" → Not in reference list → Falls through to ML model → VALID

---

## File Structure

```
ML-Data-Validator/
├── models/
│   ├── base_model.pkl        # Pre-trained base model
│   └── {custom}_model.pkl    # User's custom models
├── training_data/
│   └── base_training_data.csv  # Training data for base model
├── reference_lists/
│   ├── country.txt           # Valid countries
│   ├── age.txt               # Valid ages (0-120)
│   ├── email_domains.txt     # Common email domains
│   └── phone_prefixes.txt    # Country phone codes
└── ml/
    ├── validator.py          # UnifiedMLValidator class
    ├── corrector.py          # Correction suggestions
    └── feature_extractor.py  # 67-feature extraction
```

---

## Quick Reference

### Improve Base Model:
1. Edit `training_data/base_training_data.csv`
2. Edit `reference_lists/*.txt`
3. Retrain with Python script or restart app

### Improve Custom Model:
1. Use "Add data to existing model" in UI
2. Upload more valid examples
3. Model automatically improves

### Check What's Trained:
```python
from ml.validator import UnifiedMLValidator
validator = UnifiedMLValidator('models/base_model.pkl')
print(f"Columns: {validator.columns}")
print(f"Name whitelist size: {len(validator.column_whitelists.get('name', set()))}")
```
