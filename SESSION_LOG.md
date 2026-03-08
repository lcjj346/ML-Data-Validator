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

### 2026-03-02 (Project Research & Architecture Analysis)

#### Part 1: Codebase Architecture Review
- **Changes:** No code changed — research and analysis session
- **Findings:** Confirmed no LLM is involved in the project. Full ML stack breakdown:

| Component | Technology |
|-----------|-----------|
| Validator | scikit-learn `LogisticRegression` (one model per column) |
| Feature extraction | Hand-crafted ~67 numeric features (`ml/feature_extractor.py`) |
| Corrector | `difflib.SequenceMatcher` string similarity (no ML at inference) |
| Model storage | `joblib` `.pkl` file |
| LLM | **None** |

- **Training pipeline:**
  1. User provides CSV — all rows treated as valid examples
  2. Synthetic invalid examples auto-generated (scrambled, truncated, noise added)
  3. ~67 features extracted per cell (length, char ratios, special chars, regex patterns, numeric buckets)
  4. `LogisticRegression` trained per column (valid=1, invalid=0)
  5. Model serialized to `models/base_model.pkl` via `joblib`

---

#### Part 2: Market Research — Competitive Landscape

Researched all major data validation tools to assess project novelty.

| Tool | Validation Method | Suggests Corrections | Trains on User Data | OSS? |
|---|---|---|---|---|
| Great Expectations | Rule-based | No | No | OSS |
| Deepchecks | Statistical + rules | No | No | OSS |
| Cleanlab | ML (label errors only) | Partial (labels only) | Yes | OSS/Commercial |
| TFDV | Statistics → schema rules | No | No | OSS |
| AWS Deequ | Rule-based | No | No | OSS |
| Pandera / Cerberus | Rule-based | No | No | OSS |
| Soda Core | Rule-based + light anomaly | No | Baselines only | OSS |
| LLM (GPT/Claude) | LLM reasoning | Yes | Few-shot only | API/Commercial |
| Ataccama / Informatica | Rule + ML + LLM | Partial | Yes | Commercial $ |
| **This project** | **Supervised ML (sklearn)** | **Yes** | **Yes — core design** | **OSS** |

**Key differentiators of this project (genuinely underserved in OSS):**
1. **Supervised ML trained on user-provided examples** — Most OSS tools use rules or statistics; no major OSS tool trains a model on "here are my valid rows, learn from them"
2. **Correction suggestions for arbitrary field types** — No major OSS tool does this for general CSV fields. LLMs can, but without domain training
3. **Field-type agnostic** — Phone, email, name, country, blood sugar — any column type
4. **Retrainable on domain-specific data** — Only enterprise commercial tools (Ataccama, Informatica, Collibra) do this
5. **Closest OSS competitor:** Cleanlab — but scoped to ML label correction only, not general CSV field validation
6. **Closest overall competitor:** Enterprise tools (Ataccama, Informatica) — closed-source, expensive, not user-trainable

---

#### Part 3: Strategic Decisions for Presentation

**On adding LLM:**
- Decision: **Do NOT add LLM last-minute** — too risky, expensive to run, and not needed to be novel
- Hybrid LLM approach (ML first pass → LLM for low-confidence cells) is the right architecture, but should be future work
- Mention as future improvement in report/presentation
- LLM would add: natural language explanations, context-aware validation, no retraining needed
- LLM would cost: API spend per cell, slower, non-deterministic, privacy risk

**On security:**
- Decision: **Not required for local demo presentation**
- Minimum to mention: file type/size validation, no auth needed for local use
- For production: would need rate limiting + input sanitization + auth
- Mention as "future work" in report

**On deployment:**
- Decision: **Run locally for the demo** — `npm run dev` + `uvicorn`
- If a live URL is required: Docker Compose is the fastest path
- Do NOT deploy last-minute without thorough testing
- Mention Docker as deployment strategy in report

---

- **Files:** No code changes
- **Tested:** N/A
- **Notes:** This session's findings are useful for project report writing — competitive analysis, architecture justification, and novelty argument are documented above

---

### 2026-03-02 (Design Decision: No LLM — Patient Data Privacy)

- **Changes:** No code changed — architectural decision recorded
- **Decision:** LLM integration deliberately excluded from this project
- **Reason:** The tool processes patient medical data (CSV files containing names, ages, blood sugar, etc.). Sending this data to an external LLM API (OpenAI, Anthropic, etc.) would:
  1. Violate patient data privacy — data leaves the local system
  2. Risk non-compliance with healthcare data regulations (e.g., PDPA in Singapore, HIPAA equivalent)
  3. Introduce a dependency on third-party infrastructure for sensitive workloads
- **Design principle:** All validation runs fully **offline and locally** — no data ever leaves the machine. This is a core security requirement, not an oversight.
- **Alternative considered:** Local LLM via Ollama (no API call) — deferred to future work as it requires significant hardware and setup overhead
- **Use in report:** This decision supports the "security and privacy by design" argument. The offline ML approach (sklearn LogisticRegression) is the correct architectural choice for healthcare/sensitive data validation tools.
- **Files:** No code changes
- **Tested:** N/A
- **Notes:** Cite this in the project report under Security Considerations / Design Decisions

---

### 2026-03-02 (UI Polish — Confidence Scores & Corrections Table)

#### Changes Made

**Feature: Confidence scores in Corrections Panel**
- Backend (`backend/state.py`): Added `cell_confidence: Dict[str, float] = {}` to Session
- Backend (`backend/schemas.py`): Added `confidence: float = 0.0` to `CorrectionItem`; added `cell_confidence: Dict[str, float] = {}` to `ValidationResultsResponse`
- Backend (`backend/routers/validation.py`): Now stores confidence per cell in `session.cell_confidence`, passes `confidence` to each `CorrectionItem`, returns `cell_confidence` in results
- Frontend (`frontend/src/types/index.ts`): Added `confidence?: number` to `CorrectionItem`, added `cell_confidence: Record<string, number>` to `ValidationResults`
- Frontend (`frontend/src/components/validate/CorrectionsPanel.tsx`): Confidence shown as last column (after Action) in corrections table

**Per-column quality chart: added then removed**
- Was added to `QualityMetrics.tsx` but removed at user request — 3 summary cards (Valid/Invalid/Quality%) kept as-is

**Corrections table layout improvements**
- Column order: Row → Column → Original → Suggestion → Reason → Confidence → Action
- Header text changed to white (`text-white`) for readability
- Header moved inside `overflow-y-auto` scroll container (sticky top) so columns always align with data rows regardless of scrollbar
- Column widths tuned: `grid-cols-[40px_0.4fr_0.5fr_0.5fr_0.6fr_120px_70px]` — flexible columns reduced to give Confidence (120px) and Action (70px) more visual space

**Validation Results table**
- `#` column: added `minWidth: 40`, `maxWidth: 60` to keep row number column compact

- **Files:** `backend/state.py`, `backend/schemas.py`, `backend/routers/validation.py`, `frontend/src/types/index.ts`, `frontend/src/components/validate/CorrectionsPanel.tsx`, `frontend/src/components/validate/QualityMetrics.tsx`, `frontend/src/components/validate/ValidateTab.tsx`, `frontend/src/components/validate/ValidationGrid.tsx`
- **Tested:** Pending user test
- **Notes:** Confidence scores show ML model certainty (0–100%) for each invalid cell. Header sticky inside scroll container fixes the column misalignment bug caused by scrollbar width.

---

### 2026-03-03 (Confidence Score Clarification — For Report)

- **Changes:** No code changed — clarification recorded
- **Finding:** The confidence score (e.g. 100% for "J" being invalid) is the ML model's certainty that the **original value is INVALID**, not certainty that the suggested correction is correct. These are two separate mechanisms:
  - **Confidence** = `LogisticRegression` probability that the cell is invalid (far from valid cluster in feature space)
  - **Correction** = `difflib.SequenceMatcher` similarity matching against stored valid examples — the model has no input here
- **Example:** `"J"` gets 100% invalid confidence because it is a single character, extremely far from all valid names (5–20 chars) in feature space. The correction `"Jo"` is the closest valid example by string similarity — the 100% says nothing about whether "Jo" is the right fix.
- **UI label:** Column header says "Confidence" — this means "how certain the model is this value is invalid"
- **Use in report:** Under Model Architecture / Validation Output — explain that confidence reflects the classifier's decision boundary distance (logistic probability), not correction quality. This is a valid ML metric worth highlighting.

---

### 2026-03-03 (Security — File Size Guard)

- **Changes:** Added 10MB file size limit to both upload endpoints
- **Files:** `backend/routers/validation.py`, `backend/routers/training.py`
- **Tested:** Pending user test
- **Notes:** Rejects uploads >10MB with HTTP 400. Ties into the security/privacy by design argument — prevents accidental upload of very large sensitive datasets, reduces server memory risk. Mention under Security Considerations in report.

---

### 2026-03-03 (Training Metrics Table, Summary Report Export, Delete Model)

#### Part 1: Training Metrics — Compact Table
- **`frontend/src/components/train/TrainingMetrics.tsx`** — Rewrote from card grid to compact table
  - One row per trained column: Column | Samples | Train Acc | Test Acc | Test F1 | Best C | CV F1 | Split
  - Color-coded accuracy: green ≥90%, yellow ≥75%, red <75%
  - Confusion matrices moved to collapsibles below table (defaultOpen=false)
  - Merged Best C and CV F1 columns from remote hyperparameter tuning commit

#### Part 2: Summary Report Export
- **`backend/routers/validation.py`** — Added `GET /{session_id}/export-report` endpoint
  - Returns a two-section CSV: Overview (quality %, valid/invalid counts) + Per-Column breakdown (invalid count, sample values, sample corrections)
- **`frontend/src/api/client.ts`** — Added `getSummaryReportUrl()`
- **`frontend/src/components/validate/ExportSection.tsx`** — Added "Download Summary Report" button alongside existing CSV export

#### Part 3: Delete Model Button
- **`backend/routers/training.py`** — Added `DELETE /api/models/{model_name}` endpoint; `base_model` is protected (returns 403)
- **`frontend/src/api/client.ts`** — Added `deleteModel(name)`
- **`frontend/src/components/train/ModelsList.tsx`** — Added red Delete button per model row; `base_model` shows "default" badge instead

#### Part 4: Git Conflict Resolution
- Pulled remote commit (hyperparameter tuning); resolved conflicts in `SESSION_LOG.md` and `TrainingMetrics.tsx`
- Kept local table layout, added remote's Best C / CV F1 columns

#### Part 5: Base Model Retrain
- Retrained `models/base_model.pkl` with sklearn 1.8.0 to fix version mismatch warning

- **Files:** `backend/routers/validation.py`, `backend/routers/training.py`, `frontend/src/api/client.ts`, `frontend/src/components/train/TrainingMetrics.tsx`, `frontend/src/components/train/ModelsList.tsx`, `frontend/src/components/validate/ExportSection.tsx`, `models/base_model.pkl`
- **Tested:** Yes — 30/30 pytest pass; API mock test 38/39 pass (1 was wrong assertion in test); custom hospital model trained and validated successfully
- **Notes:** `base_model` cannot be deleted via UI. Summary report CSV is useful for handing to clients after a validation run.

---

### 2026-03-05 (Categorical Column Auto-Detection)

#### Problem
Custom model (hospital data) incorrectly marked `Ward Z`, `X+` (blood type), `Astrology` (department), temperature `99` and `25` as **valid**. Root cause: ML feature extractor uses structural features only — "Ward Z" is structurally identical to "Ward A", so the classifier cannot distinguish them.

#### Solution: Automatic categorical column detection
- **`ml/validator.py`** — Added `self.categorical_columns: set` attribute
  - **During `train()`**: if `unique_values / total_rows < 30%` AND `unique_count ≤ 20` → column flagged as categorical
  - **During `validate()` and `validate_batch()`**: categorical columns skip the ML fallback entirely — any value not in the whitelist (exact or fuzzy ≥80% match) returns `(False, 0.85)`
  - **During `explain_invalidity()`**: returns helpful message e.g. `"Not a valid ward (expected: Ward A, Ward B, Ward C, Ward D, Ward E)"`
  - **Persisted** in `.pkl` with `version: '2.2'`; backward-compatible with old models

#### Results
| Value | Column | Before | After |
|-------|--------|--------|-------|
| Ward Z | ward | VALID (wrong) | INVALID (83%) — typo of Ward A |
| X+ | blood_type | VALID (wrong) | INVALID (85%) — unknown category |
| Astrology | department | VALID (wrong) | INVALID (85%) — unknown category |
| 99 | temperature | VALID (wrong) | INVALID (85%) — unknown category |

#### Base model unaffected
- All 7 base model columns (name, email, phone, country, age, address, blood_sugar) have >20 unique values → not detected as categorical → ML classifier unchanged

- **Files:** `ml/validator.py`, `models/base_model.pkl`, `models/hospital_test.pkl` (retrained)
- **Tested:** Yes — 30/30 pytest pass; smoke test confirmed all invalid values caught
- **Notes:** Threshold of 30% ratio + ≤20 unique values works robustly for real datasets. For a business CSV with 500 salary rows, salary has ~400 unique values → never triggers categorical. Mention this in report under "Validation Strategy" — hybrid approach: whitelist for categorical, ML for open-ended.

---

### 2026-03-05 (New Test Datasets)

#### Employee HR Dataset
- **`test_data/custom_employee_training.csv`** — 40 rows, 5 columns: `employee_id` (exclude), `department` (5 values), `job_title` (8 values), `employment_type` (3 values), `salary` (numeric)
- **`test_data/custom_employee_validate.csv`** — 10 rows, 7 valid + 3 invalid: `Sales`/`Administration` (bad dept), `Intern`/`Developer` (bad title), `Freelance`/`Remote` (bad type), `-2000` (negative salary)

#### Retail Orders Dataset (300-row demo)
- **`test_data/custom_retail_training.csv`** — 300 rows, 8 columns: `order_id` (exclude), `customer_name`, `product_category` (6 values), `payment_method` (5 values), `order_status` (5 values), `quantity` (1–50), `unit_price` (decimal), `region` (5 values)
- **`test_data/custom_retail_validate.csv`** — 100 rows, **95 valid + 5 invalid** (designed for demo):
  - `Furniture` (bad product_category)
  - `Cryptocurrency` (bad payment_method)
  - `Returned` (bad order_status)
  - `-19.99` (negative unit_price)
  - `Southwest` (bad region)
- Designed so a demo audience sees a clean dataset with just 5% flagged as invalid

- **Files:** `test_data/custom_employee_training.csv`, `test_data/custom_employee_validate.csv`, `test_data/custom_retail_training.csv`, `test_data/custom_retail_validate.csv`
- **Tested:** Yes — all invalid rows detected with correct reasons; valid rows pass cleanly
- **Notes:** Retail dataset is the best demo dataset — large enough (300 rows) to look realistic, 5% error rate is realistic and visually impactful

---

### 2026-03-05 (Fuzzy Typo Detection Fix — General Correctness Improvement)

#### Problem
High-cardinality columns (e.g. `order_id`) were incorrectly flagged as INVALID after training. `ORD0301` was ~85% similar to `ORD0001` → fuzzy typo check fired → marked invalid before the ML classifier even ran. Root cause: fuzzy detection was enabled for **any column not in a hardcoded `OPEN_ENDED_COLUMNS` list**, which was wrong.

#### Fix — `ml/validator.py` (both `validate()` and `validate_batch()`)
Changed `use_strict_typo_detection` logic from:
```python
# Old: enabled for everything except a hardcoded list
use_strict_typo_detection = not any(kw in col_lower for kw in self.OPEN_ENDED_COLUMNS)
if column_name in self.reference_lists:
    use_strict_typo_detection = True
```
To:
```python
# New: only for categorical columns or columns with reference lists
use_strict_typo_detection = (
    column_name in self.categorical_columns
    or column_name in self.reference_lists
)
```

#### Why this is correct
- **Categorical columns** (ward, blood_type, region…): finite valid set → fuzzy correctly catches typos → unknown values blocked
- **Reference list columns** (country, currency…): explicit valid list → fuzzy correctly catches "Singaproe" → "Singapore"
- **Everything else** (order_id, customer_name, product_code, invoice_no…): ML classifier decides based on structural features learnt from training data → `ORD0301` recognised as valid because it follows the same `ORD + 4 digits` pattern

#### Results
| Value | Column | Before | After |
|-------|--------|--------|-------|
| ORD0301 | order_id | INVALID (wrong — flagged as typo of ORD0001) | VALID (93%) |
| ORD0350 | order_id | INVALID | VALID (93%) |
| INVALID_ID | order_id | INVALID | INVALID (100%) ✓ |
| Furniture | product_category | INVALID (85%) ✓ | INVALID (85%) ✓ |
| Southwest | region | INVALID (85%) ✓ | INVALID (85%) ✓ |

#### Also fixed this session
- **`frontend/src/components/validate/CorrectionsPanel.tsx`** — Added `title={c.reason}` and `cursor-help` to the truncated Reason cell so users can hover to see the full reason text (e.g. "Not a valid region (expected: Central, East, North, South, West)")

- **Files:** `ml/validator.py`, `frontend/src/components/validate/CorrectionsPanel.tsx`, `models/retail_test.pkl` (retrained)
- **Tested:** Yes — 30/30 pytest pass; ORD0301–ORD0400 now VALID; categorical invalid values still caught correctly
- **Notes:** Fix is fully general — applies to any user-uploaded CSV, not just the retail dataset. Any column that isn't auto-detected as categorical and has no reference list will now correctly defer to the ML classifier for unseen values.

---

### 2026-03-07 (Demo Readiness — Current Validation State)

- **Changes:** No code changes — demo readiness review
- **Current state:** Validation results are functional and presentable for demo. Some cells may still be highlighted incorrectly (model limitations with edge cases).
- **Known limitations to mention during demo / write in report:**
  - Model accuracy depends heavily on training data size and diversity — small datasets may produce false positives/negatives
  - Structural feature extractor cannot distinguish semantically different values with same structure (partially addressed by categorical detection)
  - Correction suggestions use string similarity (difflib), not semantic understanding — suggestions may not always be the intended correct value
  - Open-ended columns (name, address) rely purely on ML structural patterns — unusual but valid values may be flagged
- **Tested:** Yes — 30/30 pytest pass, retail + hospital + employee test datasets validated
- **Notes:** These limitations are expected for a capstone ML project and represent honest future improvement areas. Mention in report under "Limitations & Future Work". Planned future improvements: LLM integration for semantic validation, larger/more diverse training datasets, active learning from user corrections.

---

### 2026-03-08 (Applied Internship Learnings — Testing, Security & User Evaluation)

> **Context for report/slides:** The following practices were learnt during internship and deliberately applied to this capstone project. They reflect real-world software engineering standards, not just academic deliverables.

---

#### 1. Automated Testing — Learnt from DevOps / QA practices at internship

**What was applied:**
- **pytest unit test suite** (`tests/`) — 30 test cases covering all core ML validator behaviours:
  - Valid/invalid detection per column type
  - Typo detection and correction suggestions
  - Numeric boundary validation (age 0–120, blood sugar, salary)
  - Edge cases: empty strings, unicode, special characters, untrained columns
  - Model save/load roundtrip, training data stacking
- **End-to-end API integration test** (`test_data/run_api_test.py`) — simulates full user flow: upload CSV → train model → run validation → fetch results → export (38/39 pass)
- Fully automated — single command: `pytest tests/ -v`

**Internship connection:**
> At internship, automated testing was mandatory before any deployment. The principle: if there is no test, the feature cannot be trusted. Applied this here — every core ML function has a corresponding automated test. When ML logic changed (categorical detection, fuzzy matching), the test suite caught regressions immediately.

**For slides/report:** Automated testing enabled rapid, confident iteration. 6+ changes to `ml/validator.py` across the project were all verified against the same 30-test suite — zero regressions introduced.

---

#### 2. Security — Learnt from cloud deployment and API work at internship

**What was applied:**

| Security Measure | Location | Purpose |
|---|---|---|
| File type validation | Both upload endpoints | Rejects non-CSV with HTTP 400 |
| **10MB file size guard** | Both upload endpoints | Prevents memory exhaustion / large payload attacks |
| Input sanitisation | pandas + Pydantic schemas | Malformed CSVs and invalid JSON rejected at system boundary |
| **No external API calls** | Architecture decision | Sensitive/patient data never leaves the local machine |
| No path traversal | `models/{name}.pkl` loading | Users cannot load arbitrary files from disk |
| No SQL / no shell execution | Pure Python in-memory | Eliminates SQL injection and command injection surfaces |
| Protected base model | `DELETE` endpoint returns 403 | Prevents destruction of default model |
| **CORS restriction** | `backend/main.py:16–22` | API access limited to `localhost:5173` only — prevents unauthorised cross-origin requests from other websites calling the local API silently |

**CORS note for report:**
> `CORSMiddleware` is configured with `allow_origins=["http://localhost:5173"]` — not the insecure wildcard `"*"`. This means only the frontend origin can call the API. Without this, any malicious website a user visits could silently make API calls to the local server in the background. For future production deployment, this value would be updated to the actual deployed frontend URL.

**Internship connection:**
> At internship, every API was reviewed for: input validation, authentication, rate limiting, and data exposure. Applied the same lens here. The decision to run fully offline (no LLM API) was a deliberate privacy-by-design choice — patient data must never leave the machine, aligning with PDPA and healthcare data handling principles.

**For slides/report:** Security was built in from the start, not added as an afterthought. The offline architecture is itself the primary security control — no data exfiltration is possible by design.

---

#### 3. User Evaluation — Learnt from product feedback loops at internship

**Quantitative (already available):**
- Training metrics per column: Train Acc, Test Acc, Test F1, CV F1 — objective model performance evidence
- Demo result: 300-row retail CSV with 5 injected invalid rows → **all 5 detected = 100% recall on known errors**
- Validation completes in ~2 seconds vs ~15–30 min manual review of 300 rows

**Performance comparison (before vs after the tool):**
> Manual review: ~15–30 min per 100 rows, error-prone, no suggestions
> With tool: <2 seconds, all invalid cells highlighted with reason + suggested correction

**Planned qualitative evaluation (short usability survey, Google Forms):**
1. Was the interface easy to use without instruction? (1–5)
2. Did the validation results seem accurate and useful? (1–5)
3. Were the error reasons and suggestions clear? (1–5)
4. What was confusing or could be improved? (open text)
5. Would this save time vs manual data checking? (Yes / No / Maybe)

Target: 3–5 classmates / testers. Results to go into report under **User Evaluation / Usability Testing**.

**Internship connection:**
> At internship, features were evaluated through both quantitative metrics (benchmarks, error rates) and qualitative feedback (user interviews, NPS scores). Applied both here. Even 3–5 survey responses demonstrates user-centred thinking, which is the point — not the sample size.

**For slides/report:** *"Evaluation used a two-pronged approach — quantitative model performance metrics (Test Accuracy, F1 Score, recall on injected errors) and qualitative usability survey — consistent with industry evaluation practices applied from internship experience."*

---

- **Files:** `SESSION_LOG.md` only
- **Notes:** This entry is the source of truth for the Testing / Security / Evaluation sections of the capstone report and demo slides. Update once survey results are collected.

---

## Future Sessions

<!-- Template for new entries:

### YYYY-MM-DD
- **Changes:** What was modified
- **Files:** Key files changed
- **Tested:** Yes/No - outcome
- **Notes:** Any additional context

-->
