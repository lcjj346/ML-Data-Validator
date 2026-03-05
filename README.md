# ML Data Validator

Train ML models on your data, validate any column type, and get correction suggestions.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for frontend)

## Setup (First Time / New Computer)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install frontend dependencies and build

```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Run the app

```bash
python run.py
```

Open **http://localhost:8000** in your browser.

## Development Mode

For active development with hot-reload on frontend changes:

**Terminal 1** - Backend:
```bash
python run.py
```

**Terminal 2** - Frontend dev server:
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** (Vite proxies API calls to the backend).

## Usage

### Train a Model

1. Go to the **Train Models** tab
2. Upload a CSV file (all rows are treated as valid examples)
3. Optionally exclude columns (like ID, timestamp)
4. Choose "Create new model" and give it a name
5. Click **Train Model**
6. View per-column accuracy, F1 score, and confusion matrix

### Validate Data

1. Go to the **Validate Data** tab
2. Upload a CSV file to validate
3. Select a trained model
4. Click **Validate**
5. Results:
   - **Green cells** = Valid
   - **Red cells** = Invalid
   - **Orange cells** = Manually edited
6. Review suggested corrections and apply individually or all at once
7. Export the validated CSV

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
ML-Data-Validator/
├── backend/                     # FastAPI backend
│   ├── main.py                  # App entry, CORS, static files
│   ├── state.py                 # In-memory session store
│   ├── schemas.py               # Pydantic request/response models
│   └── routers/
│       ├── validation.py        # Upload, validate, correct, export
│       └── training.py          # Upload, train, list models
│
├── frontend/                    # React + TypeScript + Vite
│   └── src/
│       ├── App.tsx              # Tab navigation
│       ├── api/client.ts        # API client (fetch + SSE)
│       ├── types/index.ts       # TypeScript interfaces
│       ├── hooks/               # useValidation, useTraining, useToast
│       └── components/
│           ├── validate/        # Validate tab components
│           └── train/           # Train tab components
│
├── ml/                          # Core ML code (unchanged)
│   ├── validator.py             # Unified ML validator
│   ├── corrector.py             # Similarity-based corrector
│   └── feature_extractor.py     # 71-feature extractor
│
├── models/                      # Trained .pkl models
├── training_data/               # Training CSV files
├── reference_lists/             # Valid values for countries, etc.
├── test_data/                   # Sample test CSVs
├── tests/                       # Pytest test suite (30 tests)
├── run.py                       # Entry point (uvicorn)
└── requirements.txt             # Python dependencies
```

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Frontend**: React + TypeScript + Vite + Tailwind CSS
- **Table**: AG Grid (Community Edition) with cell coloring
- **ML**: scikit-learn (Logistic Regression) + pandas + numpy
- **Progress**: Server-Sent Events (SSE) for real-time updates
- **Deployment**: Single server - FastAPI serves built React static files
