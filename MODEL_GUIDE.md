# ML Data Validator - Model Guide

## How Validation Works

When validating a cell value, the system follows this pipeline:

```
1. Whitelist exact match (case-insensitive)
   └─ Found? → VALID (100%)

2. Numeric normalisation
   └─ 95 == 95.0 == "95"? → VALID (100%)

3. Fuzzy typo detection  [categorical & reference-list columns only]
   └─ ≥80% similar to a valid example? → INVALID (shows correction)
   └─ High-cardinality columns (order_id, names) skip this step entirely

4. Hardcoded boundary rules
   └─ Negative age / salary, age > 120, etc. → INVALID with reason

5. ML classifier  [open-ended & high-cardinality columns]
   └─ 67-feature extraction → LogisticRegression prediction → confidence score
```

---

## Two Model Types

### Base Model (`models/base_model.pkl`)
- Pre-trained on common columns: `name`, `email`, `phone`, `country`, `age`, `address`, `blood_sugar`
- Ships with the app, ready to use immediately
- Retrain by updating `training_data/base_training_data.csv` (see below)

### Custom Models (`models/{name}.pkl`)
- Train on any CSV domain via the **Train** tab in the UI
- All rows in the uploaded CSV are treated as valid examples
- Synthetic invalid examples are auto-generated for training

---

## Categorical Column Auto-Detection

During training, any column where:
- `unique_values / total_rows < 30%` **AND**
- `unique_value_count ≤ 20`

…is automatically flagged as **categorical** and gets whitelist-only validation. This means:
- Only exact matches (or ≥80% fuzzy matches) to training values are accepted as valid
- The ML classifier is bypassed entirely for these columns
- Unknown values return: `"Not a valid {column} (expected: val1, val2, ...)"`

**Examples:**
- `department` (5 values / 40 rows) → categorical → `"Astrology"` correctly INVALID
- `ward` (5 values / 40 rows) → categorical → `"Ward Z"` correctly INVALID
- `salary` (38 values / 40 rows) → not categorical → ML classifier used
- `order_id` (300 values / 300 rows) → not categorical → ML classifier used

---

## Improving the Base Model

### 1. Add Training Data

Edit `training_data/base_training_data.csv`:
```csv
name,email,phone,country,age,address,blood_sugar
Jean-Pierre,jp@email.fr,+33 1 2345 6789,France,45,1 Rue de Paris,95.0
```

**Tips:**
- Add diverse examples across cultures, formats, and edge cases
- Include variations: `"USA"`, `"US"`, `"United States"`
- Add hyphenated names, compound names, single names
- Include multiple phone formats: `+65 9123 4567`, `91234567`, `(555) 123-4567`

### 2. Expand Reference Lists

Reference lists enforce strict validation for finite value sets.

| File | Purpose | Example Values |
|------|---------|----------------|
| `reference_lists/country.txt` | Valid countries | Singapore, USA, UK |
| `reference_lists/age.txt` | Valid ages | 0–120 |
| `reference_lists/email_domains.txt` | Common domains | gmail.com, yahoo.com |
| `reference_lists/phone_prefixes.txt` | Country codes | +1, +65, +44 |
| `reference_lists/gender.txt` | Gender values | Male, Female, M, F |
| `reference_lists/currency.txt` | Currency codes | USD, SGD, EUR |
| `reference_lists/status.txt` | Status values | Active, Pending, etc. |

### 3. Retrain via Python

```python
from ml.validator import UnifiedMLValidator
import pandas as pd

df = pd.read_csv('training_data/base_training_data.csv')
validator = UnifiedMLValidator()
metrics = validator.train(df, model_name='base_model')
validator.save('models/base_model.pkl')

for col, m in metrics.items():
    print(f"{col}: {m['test_accuracy']:.1%}  (best C={m.get('best_C')})")
```

---

## Training Metrics

After training, the UI displays a per-column metrics table:

| Metric | Meaning | Target |
|--------|---------|--------|
| Train Acc | % correct on training data | — |
| Test Acc | % correct on held-out split | >85% |
| Test F1 | F1 score on held-out split | >85% |
| Best C | GridSearchCV-selected regularisation | — |
| CV F1 | Cross-validation F1 score | >85% |

**Colour coding:** green ≥90%, yellow ≥75%, red <75%

**Low accuracy causes:**
- Too few training examples (aim for 50+ valid rows per column)
- Too little diversity in the training data
- Column values are random/high-cardinality (expected — ML handles these)

---

## Hyperparameter Tuning

The system automatically runs GridSearchCV over `C = [0.01, 0.1, 1.0, 10.0, 100.0]` for each column:

- **Low C** (0.01, 0.1) — heavy regularisation, good for structured columns like email
- **High C** (10, 100) — minimal regularisation, good for diverse columns like names/countries

If there are too few samples per class for cross-validation, the system falls back to `C=1.0`.

---

## Column Type Recommendations

| Column Type | Recommended Approach | Notes |
|-------------|---------------------|-------|
| Names | More training data | Add cultural variations |
| Countries | Reference list | Add all valid abbreviations |
| Emails | Training data + domain list | Patterns are learnable |
| Phone | Training data + prefix list | Add format variations |
| Age / Blood Sugar | Boundary rules apply automatically | Numeric columns |
| Departments / Wards | Auto-detected as categorical | No extra config needed |
| Order IDs / Codes | ML classifier | High cardinality, structural patterns |

---

## File Structure

```
ML-Data-Validator/
├── models/
│   ├── base_model.pkl           # Pre-trained base model (protected)
│   └── {custom}.pkl             # User-trained custom models
├── training_data/
│   └── base_training_data.csv   # 544 rows, 7 columns
├── reference_lists/
│   ├── country.txt
│   ├── age.txt
│   ├── email_domains.txt
│   ├── phone_prefixes.txt
│   ├── gender.txt
│   ├── currency.txt
│   └── status.txt
└── ml/
    ├── validator.py             # UnifiedMLValidator (train, validate, correct, explain)
    └── feature_extractor.py     # GenericFeatureExtractor (67 features)
```

---

## Quick Reference

| Task | How |
|------|-----|
| Improve base model accuracy | Add rows to `training_data/base_training_data.csv`, retrain |
| Add a new valid country | Edit `reference_lists/country.txt`, retrain |
| Train a custom model | Upload CSV in the Train tab |
| Delete a custom model | Click Delete in the model list (base_model is protected) |
| Inspect a trained model | `python -c "from ml.validator import UnifiedMLValidator; v = UnifiedMLValidator('models/base_model.pkl'); print(v.columns, v.categorical_columns)"` |
