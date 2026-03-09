# ML Data Validator

A machine learning-based data validation and correction tool. Upload a CSV, train a custom model on your data, and the system highlights invalid cells, explains why they're wrong, and suggests corrections — all running fully offline with no external API calls.

---

## Quick Start

**Backend** (terminal 1):
```bash
python run.py
```

**Frontend** (terminal 2):
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## How It Works

### Validate Tab
1. Upload a CSV file
2. Select a model (base model or a custom trained model)
3. Click **Run Validation** — invalid cells highlight red, valid cells green
4. Review suggested corrections in the panel below (with reason + confidence score)
5. Apply corrections individually or all at once
6. Export the cleaned CSV or a summary quality report

### Train Tab
1. Upload a CSV with your valid data (all rows treated as valid examples)
2. Give the model a name and click **Train**
3. The system auto-generates synthetic invalid examples and trains a Logistic Regression classifier per column
4. View per-column accuracy, F1 score, and GridSearchCV tuning results
5. Saved model appears in the model list — ready to use for validation

---

## Architecture

```
frontend/          React + Vite + TypeScript + Tailwind CSS + AG Grid
backend/           FastAPI (Python) — REST API + SSE streaming
ml/                scikit-learn ML core (no LLM, fully offline)
models/            Trained .pkl model files
training_data/     Base training CSV (544 rows, 7 columns)
reference_lists/   Valid value lists (country, gender, currency, etc.)
tests/             30 pytest unit tests
test_data/         Demo CSVs and API integration test
```

**ML validation pipeline per cell:**
1. Whitelist exact match
2. Numeric normalisation (`95` == `95.0`)
3. Fuzzy typo detection — categorical/reference-list columns only (≥80% similarity)
4. Hardcoded boundary rules (age 0–120, no negative salary, etc.)
5. Logistic Regression classifier — 67 structural features, GridSearchCV-tuned C

**Correction engine:** `difflib.SequenceMatcher` — finds the most structurally similar valid example from training data. Deterministic, <10ms, no ML at inference.

---

## Base Model Columns

The pre-trained `base_model.pkl` validates these columns (auto-matched by name):

| Column | Validation Type | Notes |
|--------|----------------|-------|
| `name` | ML classifier | Names across cultures, hyphenated |
| `email` | ML + rules | Must have `@`, valid domain |
| `phone` | ML + rules | Minimum 7 digits, various formats |
| `country` | Reference list | 143 countries + abbreviations |
| `age` | Numeric bounds | 0–120, no negatives |
| `address` | ML classifier | Structural patterns |
| `blood_sugar` | Numeric bounds | 0–500 |

---

## Custom Models

Train on any CSV domain (HR, retail, hospital, etc.). The system automatically:
- Detects **categorical columns** (e.g. `department`, `ward`, `blood_type`) and applies whitelist-only validation
- Uses the **ML classifier** for open-ended and high-cardinality columns (e.g. `order_id`, `customer_name`)
- Tunes regularisation parameter **C** via GridSearchCV over `[0.01, 0.1, 1.0, 10.0, 100.0]`

See [MODEL_GUIDE.md](MODEL_GUIDE.md) for guidance on improving accuracy.

---

## Testing

```bash
# Unit tests (30 tests)
pytest tests/ -v

# End-to-end API integration test (requires server running)
python test_data/run_api_test.py
```

---

## Demo Datasets

| File | Description |
|------|-------------|
| `test_data/custom_retail_training.csv` | 300-row retail orders training data |
| `test_data/custom_retail_validate.csv` | 100-row retail data (5 injected errors) |
| `test_data/custom_employee_training.csv` | 40-row HR employee training data |
| `test_data/custom_employee_validate.csv` | 10-row employee data (3 injected errors) |

---

## Security

- All processing runs locally — no data leaves the machine
- CORS restricted to `localhost:5173` only
- CSV-only uploads with 10MB size limit
- Pydantic schema validation on all API request bodies
- Base model protected from deletion

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, TypeScript, Tailwind CSS, AG Grid |
| Backend | FastAPI, Pydantic, uvicorn |
| ML | scikit-learn, pandas, joblib |
| Correction | Python `difflib` |
| Testing | pytest |
