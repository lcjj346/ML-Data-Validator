# ML Data Validator - Session Log

> **How to use this file:**
> After each successful change/test session, add an entry below with:
> - Date
> - What was changed
> - Files modified (if notable)
> - Test outcome / Notes
>
> Say "log it" after confirming changes work, and I'll append an entry.

---

## Session History

### 2026-02-04
- **Changes:** Set up progress tracking and sub-agent workflow system
  - Created `SESSION_LOG.md` for tracking progress
  - Created `CLAUDE.md` with project instructions (auto-read by Claude)
  - Configured 3 sub-agent workflows: Code Cleanup, Testing, Logging
  - Documented recommended workflow practices
  - Ran initial codebase scan with all 3 agents in parallel
- **Files:** `SESSION_LOG.md`, `CLAUDE.md` (both new)
- **Tested:** Yes - sub-agents ran successfully in parallel
- **Notes:**
  - Cleanup agent found critical issue: `self.typo_map` undefined in `ml/corrector.py:81`
  - Testing agent confirmed: no automated test infrastructure exists
  - Backfilled git history into session log for context

---

## Historical Commits (Backfilled from Git)

### 2026-02-02
- Improved model accuracy and UI (`d4e27f3`)

### 2025-12-04
- Updated models and readme (`8549dd8`)

### 2025-12-01
- Refactor: Remove legacy dependencies and clean up codebase (`095f41f`)

### 2025-11-30
- Updated README with expanded model documentation (20 models, new validators) (`c7227af`)

### 2025-11-29
- Fixed ValueError: X has 71 features, but LogisticRegression expecting 51 features (`5882488`)
- Merge branch sync (`adf921c`)

### 2025-11-21 - 2025-11-22
- Added more base columns (`924a20d`)
- Added company custom data (`f83f54e`)
- Fixed manual edit issue (`c04c0f5`)
- Updated readme (`0e7d6e1`)

### 2025-11-19 - 2025-11-20
- Improved training data and models (`2a6035c`)
- Add validation explanations and improve training data (`b24bfd4`)
- Fix: Clear validation state when new file uploaded (`87b9548`)

### 2025-10-28 - 2025-10-30
- Revamped project structure and code (`4ad7974`)
- Integrate NLP libraries (`0bb1356`)
- Updated training_data and fine tuned model (`c628997`)
- Refactor code, updated readme and training data (`b9e9ea4`, `515e784`)
- Highlight orange for manual edits (`faec100`)
- Update readme (`f785289`)

---

### 2026-02-05 (Major Refactor Session)

#### Part 1: Unified Model Architecture Implementation
- **Changes:** Complete rewrite of ML validation system
  - Implemented `UnifiedMLValidator` class replacing old separate validators
  - Single model file now contains all columns (previously separate models per column)
  - Training now uses ALL rows as valid examples (no labels needed)
  - Added synthetic invalid data generation for training
  - Added column auto-matching (exact, normalized, substring matching)

- **Files Modified:**
  - `ml/validator.py` - Complete rewrite with new architecture
  - `ml/feature_extractor.py` - Added optional `column_name` parameter
  - `app.py` - Simplified training/validation UI

- **New Validation Flow:**
  ```
  Whitelist → Numeric Normalization → Fuzzy Typo Detection → ML Model
  ```

#### Part 2: Typo Detection & Reference Lists
- **Changes:** Added intelligent typo detection system
  - Fuzzy matching with >80% similarity threshold
  - Selective typo detection (only for columns with reference lists)
  - This prevents false positives on open-ended columns like names

- **Reference Lists Created:** (`reference_lists/`)
  | File | Contents |
  |------|----------|
  | `country.txt` | 143 countries + abbreviations (USA, UK, SG, etc.) |
  | `age.txt` | Valid ages 0-120 |
  | `email_domains.txt` | 60 common email domains |
  | `phone_prefixes.txt` | 117 country phone codes |

- **Bugs Fixed:**
  - "Singaproe" was showing as VALID → Now INVALID with "Singapore" suggestion
  - "USA", "UK" were showing as INVALID → Now VALID (100%)
  - Numeric values like `95` vs `95.0` mismatch → Added numeric normalization

#### Part 3: Model Stacking Feature
- **Changes:** Added ability to improve models incrementally
  - New `add_training_data()` method in validator
  - UI option: "Add data to existing model"
  - Adds new values to whitelists without full retraining
  - Optional retrain for ML component

#### Part 4: Training Data Expansion
- **Changes:** Massively expanded base training data
  - Increased from ~100 rows to **544 rows**
  - Added 195+ unique names (Asian, Arabic, Hispanic, hyphenated, single names)
  - Added 149+ phone formats (various country formats, with/without spaces)
  - Added 124+ email formats (+tags, dots, subdomains)
  - Added age range 0-120
  - Added blood sugar range 3.0-350.0 (mmol/L and mg/dL units)

- **Hyphenated Names Added:**
  - French: Jean-Pierre, Marie-Claire, Jean-Paul, Anne-Sophie, etc.
  - Korean: Kim Ji-Yeon, Park Min-Jun, Lee Soo-Hyun, etc.
  - Double-barrel: Smith-Jones, Brown-Wilson, Taylor-Clark, etc.
  - American compound: Billy-Bob, Betty-Lou, Mary-Ann, Jo-Ann, etc.

#### Part 5: Bug Fixes & Cleanup
- **Fixes:**
  - Fixed `use_container_width` Streamlit deprecation warning
  - Cleared `__pycache__` to resolve import errors
  - Removed `name.txt` reference list (caused false positives on valid names)
  - Fixed blood sugar validation (5.2 mmol/L was invalid)

- **Cleanup:**
  - Removed old training files
  - Removed redundant model files
  - Killed multiple Streamlit processes (ports 8501-8506)

#### Final Test Results:
| Test Case | Before | After |
|-----------|--------|-------|
| Jean-Pierre | INVALID (66%) | VALID (100%) |
| John Smith | INVALID | VALID (96%) |
| Singaproe | VALID | INVALID (89%) → Singapore |
| USA | INVALID | VALID (100%) |
| 95 (blood_sugar) | INVALID | VALID (100%) |
| Mary-Ann | INVALID (53%) | VALID (100%) |
| Billy-Bob | INVALID (51%) | VALID (100%) |

#### Model Accuracy After Retraining:
| Column | Test Accuracy |
|--------|---------------|
| name | 90.4% |
| email | 91.7% |
| phone | 96.8% |
| country | 86.7% |
| age | 92.7% |
| address | 95.0% |
| blood_sugar | 94.5% |

- **Files:** `ml/validator.py`, `ml/feature_extractor.py`, `app.py`, `training_data/base_training_data.csv`, `models/base_model.pkl`, `reference_lists/*`
- **Tested:** Yes - comprehensive testing throughout session
- **Notes:**
  - Created `MODEL_GUIDE.md` documenting how to improve model accuracy
  - Old models incompatible with new architecture (clean break)
  - Users can now train on ANY CSV structure without manual column mapping

#### Part 6: Code Cleanup & Test Suite
- **Code Fixes:**
  - Moved `json` import from line 381 to top-level (`app.py`)
  - Moved `math` import from line 174 to top-level (`ml/feature_extractor.py`)
  - Extracted duplicate `open_ended_columns` to class constant `OPEN_ENDED_COLUMNS` (`ml/validator.py`)

- **Test Suite Created:** `tests/test_validator.py` (30 tests)
  | Test Class | Tests | Coverage |
  |------------|-------|----------|
  | TestFeatureExtractor | 7 | Feature extraction, consistency, patterns |
  | TestUnifiedMLValidator | 7 | Training, validation, save/load, corrections |
  | TestTypoDetection | 3 | Typo detection for countries vs names |
  | TestNumericValidation | 2 | Numeric normalization |
  | TestModelStacking | 2 | Adding data to existing models |
  | TestCorrector | 4 | Correction suggestions |
  | TestEdgeCases | 5 | Empty data, unicode, special chars |

- **New Files:**
  - `tests/__init__.py`
  - `tests/test_validator.py`
  - `pytest.ini`
  - `MODEL_GUIDE.md`

- **Updated:** `requirements.txt` (added pytest, pytest-cov)
- **Test Result:** All 30 tests passed

---

### 2026-02-06 (Validation Improvements & Cleanup)

#### Part 1: Bug Fixes
- Fixed `width='stretch'` errors in `app.py` (7 occurrences) → `use_container_width=True`
- Fixed sklearn version mismatch warnings - retrained base model with sklearn 1.7.0
- Fixed model naming inconsistency (`_model.pkl` → `.pkl`)
- Removed print spam ("Model loaded" on every load)
- Installed missing pytest dependencies

#### Part 2: Cleanup
- **Models:** Removed redundant `base_model_model.pkl` and `sample_data_model.pkl` (kept only `base_model.pkl`)
- **Test data:** Removed 3 confusing CSV files, created single `demo_validation_test.csv`
- **README files:** Removed extra READMEs (test_data/README.md, .pytest_cache/README.md)

#### Part 3: Improved Validation Logic
- **Hardcoded validation rules** for base model columns:
  - `age`: Must be 0-120, cannot be negative
  - `blood_sugar`: Must be 0-500, cannot be negative
  - `phone`: Minimum 7 digits required
  - `email`: Must have @, valid domain, fixed @@ typo correction
  - `percentage`: Must be 0-100
  - `salary/price/amount`: Cannot be negative

- **Smarter correction suggestions:**
  - Numeric columns (age, blood_sugar, salary): No corrections (numbers don't have typos)
  - Phone/email: Requires 85% similarity (prevents random suggestions)
  - Email @@: Now suggests fix (test@@email.com → test@email.com)

- **Better error explanations:**
  - Before: "too short" for age -5
  - After: "Age cannot be negative"

#### Part 4: New Reference Lists
- `reference_lists/gender.txt` (8 values: Male, Female, M, F, Other, etc.)
- `reference_lists/currency.txt` (37 currencies: USD, SGD, EUR, etc.)
- `reference_lists/status.txt` (24 statuses: Active, Pending, Completed, etc.)

#### Test Results
| Input | Column | Before | After |
|-------|--------|--------|-------|
| `-5` | age | "Too short" | "Age cannot be negative" |
| `150` | age | Valid | Invalid - "Age must be between 0 and 120" |
| `+65 9999` | phone | Suggested random phone | No suggestion (incomplete) |
| `test@@email.com` | email | No suggestion | `test@email.com` |
| `Singaproe` | country | Correction: Singapore | Correction: Singapore (unchanged) |

- **Files:** `app.py`, `ml/validator.py`, `test_data/demo_validation_test.csv`, `reference_lists/*.txt`
- **Tested:** Yes - 30 tests passing
- **Notes:** New columns (gender, currency, status, salary, percentage, date) will auto-validate when users upload data with those column names

---

### 2026-02-19 (Frontend UI & Cleanup)

#### Part 1: Layout & Grid Improvements
- **`frontend/src/App.tsx`** — Changed max width from `max-w-7xl` to `max-w-[95vw]` for a wider, more responsive layout
- **`frontend/src/components/validate/ValidationGrid.tsx`** — Increased AG Grid default `minWidth` from `100` to `180` for more readable column widths

#### Part 2: AG Grid Header Text Fix
- **`frontend/src/index.css`** — Fixed invisible/dim header text in AG Grid v35 dark theme:
  - Added `--ag-header-foreground-color: #f3f4f6`
  - Added `.ag-header-cell-text` override: `color: #f3f4f6 !important; font-weight: 600 !important`

#### Part 3: Cleanup
- **`app.py`** — Deleted legacy Streamlit app (project fully migrated to React frontend)
- **`requirements.txt`** — Removed legacy `streamlit>=1.28.0` and `streamlit-aggrid>=0.3.4` dependencies
- **`models/base_model.pkl`** — Retrained base model

- **Files:** `frontend/src/App.tsx`, `frontend/src/components/validate/ValidationGrid.tsx`, `frontend/src/index.css`, `requirements.txt`, `models/base_model.pkl`, `app.py` (deleted)
- **Tested:** Yes - inferred from committed changes
- **Notes:** Project now fully decoupled from Streamlit; React frontend is the sole UI

---

### 2026-02-20 (Logistic Regression Tuning)

#### Part 1: Hyperparameter Tuning via GridSearchCV
- **`ml/validator.py`** — Replaced fixed `LogisticRegression(C=1.0)` with `GridSearchCV` tuning over `C = [0.01, 0.1, 1.0, 10.0, 100.0]`
  - Adaptive CV folds: up to 5, bounded by smallest class count per column
  - Falls back to default C=1.0 if not enough samples per class for CV
  - Stores `best_C` and `cv_f1_score` in training metrics per column

#### Part 2: Frontend Metrics Display
- **`frontend/src/types/index.ts`** — Added `best_C?: number` and `cv_f1_score?: number` to `ColumnMetrics`
- **`frontend/src/components/train/TrainingMetrics.tsx`** — Added "Hyperparameter Tuning (GridSearchCV)" section showing Best C and CV F1 Score after training

#### Part 3: Base Model Retrained
- Retrained `models/base_model.pkl` with tuned C values

#### Tuning Results:
| Column | Best C | CV F1 | Test Accuracy |
|--------|--------|-------|---------------|
| name | 100.0 | 88.2% | 85.8% |
| email | 0.1 | 95.5% | 93.6% |
| phone | 10.0 | 95.8% | 97.2% |
| country | 100.0 | 89.7% | 86.7% |
| age | 10.0 | 92.5% | 92.2% |
| address | 1.0 | 95.7% | 94.0% |
| blood_sugar | 10.0 | 96.2% | 97.2% |

#### Part 4: README Cleanup
- **`README.md`** — Removed stale reference to deleted `app.py`

- **Files:** `ml/validator.py`, `frontend/src/types/index.ts`, `frontend/src/components/train/TrainingMetrics.tsx`, `models/base_model.pkl`, `README.md`
- **Tested:** Pending - run `python run.py` + `cd frontend && npm run dev` to verify tuning metrics display in Train tab
- **Notes:** email performs best with heavy regularization (C=0.1 — strict patterns); name/country need almost none (C=100 — highly varied data)

---

## Future Sessions

<!-- Template for new entries:

### YYYY-MM-DD
- **Changes:** What was modified
- **Files:** Key files changed
- **Tested:** Yes/No - outcome
- **Notes:** Any additional context

-->
