# Capstone Project Final Report
## ML Data Validator — Machine Learning-Based Data Validation and Correction System

**ICT4011 Capstone Project**
**Organisation:** AMILI Pte Ltd
**Project Period:** October 2025 – March 2026

---

## Table of Contents

1. Introduction
   - 1.1 Brief Overview of Internship
   - 1.2 Project Goal and Relevance
   - 1.3 Summary of Developments Since Progress Report

2. Background Research
   - 2.1 Problem Domain and Industry Context
   - 2.2 Review of Academic Solutions
   - 2.3 Review of Open-Source Tools
   - 2.4 Market Gap Analysis

3. Requirements and System Design
   - 3.1 Functional Requirements
   - 3.2 Non-Functional Requirements
   - 3.3 Final System Architecture
   - 3.4 Architecture Decision: From Streamlit to React + FastAPI

4. Implementation Details
   - 4.1 Project Structure and Code Organisation
   - 4.2 ML Core: Feature Extraction
   - 4.3 ML Core: Validation Pipeline and Categorical Detection
   - 4.4 ML Core: Correction Engine
   - 4.5 Training Pipeline and Hyperparameter Tuning
   - 4.6 Backend API (FastAPI)
   - 4.7 Frontend (React + TypeScript)
   - 4.8 Security Implementation

5. Testing and Evaluation
   - 5.1 Automated Unit Testing (pytest)
   - 5.2 API Integration Testing
   - 5.3 Model Performance Evaluation
   - 5.4 Demo Dataset Validation Results
   - 5.5 User Evaluation

6. Technical Challenges and Adaptations
   - 6.1 Data Privacy and the No-LLM Architectural Decision
   - 6.2 Categorical vs Open-Ended Column Validation
   - 6.3 Fuzzy Typo Detection Generalisation
   - 6.4 Architecture Migration: Streamlit to Full-Stack

7. Application of Knowledge
   - 7.1 Classroom Knowledge Applied
   - 7.2 Internship Knowledge Applied
   - 7.3 Self-Directed Learning

8. Project Management
   - 8.1 Development Timeline and Milestone Progression
   - 8.2 Version Control and Session Logging
   - 8.3 Iterative Development Process

9. Limitations and Future Work

10. Conclusion and Reflections
    - 10.1 Key Technical Learnings
    - 10.2 Professional Development
    - 10.3 Personal Growth

11. References

---

## 1. Introduction

### 1.1 Brief Overview of Internship

As part of my third-year Software Engineering programme at Singapore Institute of Technology, I undertook a capstone internship at AMILI Pte Ltd, Southeast Asia's first precision gut microbiome company. During this internship, I was directly exposed to real-world data quality challenges within a healthcare research environment, specifically around the management of donor survey data pipelines. These pipelines collect and process large volumes of participant health data — including demographic information, dietary habits, and medical measurements — where data accuracy is foundational to research validity and operational efficiency.

At AMILI, data entry errors are a recurring operational challenge. Survey responses frequently contain formatting inconsistencies, impossible values (such as an age of 200 or a negative blood sugar reading), and typographical mistakes (such as "Singaproe" instead of "Singapore") that must be identified and corrected before the data enters analysis pipelines. At the time of my internship, this process was largely manual, time-consuming, and error-prone. This experience directly motivated the development of an automated, machine learning-based solution for data validation and correction.

### 1.2 Project Goal and Relevance

The goal of this project is to develop a versatile, machine-learning-based data validation and correction tool that can be used across industries and data types. Rather than building a rigid rule-based validator for a single fixed schema, the system is designed to learn what "valid" data looks like from user-provided examples, and then flag and suggest corrections for anomalies in new data.

The primary use case is based on AMILI's healthcare data challenges — validating fields such as phone numbers, ages, blood sugar levels, countries, and names. However, the tool is deliberately designed to be domain-agnostic: the same system can be trained on retail order data, HR employee records, or any structured CSV dataset. The user provides their own labelled examples and the system adapts accordingly.

A key design constraint was the decision not to use any external Language Model API (such as OpenAI or Anthropic). Since the system processes potentially sensitive patient and donor data, sending that data to a third-party cloud service would risk violating Singapore's Personal Data Protection Act (PDPA) and healthcare privacy norms at AMILI. The solution therefore runs entirely offline and locally — no data ever leaves the user's machine. This privacy-by-design constraint shaped every architectural decision made throughout the project.

The final system consists of a full-stack web application with a React frontend (TypeScript, Vite, Tailwind CSS) and a FastAPI backend (Python), with a scikit-learn machine learning core. Users can upload a CSV file, train a custom validator model on their data, run validation to highlight invalid cells, review suggested corrections, manually edit values, and export the cleaned dataset — all through a browser-based interface without any local installation.

### 1.3 Summary of Developments Since Progress Report

The Progress Report (Form E1) documented the system as it stood at Milestone 2, where the frontend was built using Streamlit and the architecture was a single-layer Python application. Since that submission, the project has undergone significant development across multiple dimensions:

**Architecture Migration**: The most substantial change was the complete replacement of the Streamlit frontend with a modern full-stack architecture — a React (Vite + TypeScript) frontend communicating with a FastAPI backend. This decoupling enabled a significantly richer user interface and a cleaner separation of concerns between UI logic and business logic.

**ML Pipeline Improvements**: Hyperparameter tuning using GridSearchCV was added to automatically optimise the regularisation parameter C for each column's Logistic Regression classifier. A new categorical column auto-detection mechanism was also implemented, which identifies columns with finite valid value sets (such as blood type or ward number) and applies whitelist-only validation instead of the general ML classifier.

**Validation Accuracy**: The fuzzy typo detection logic was fixed to only fire for categorical columns and reference-list columns, preventing high-cardinality columns (such as order IDs) from being incorrectly flagged as typos of training examples.

**Quality and Security**: A 30-test automated pytest suite was created alongside an end-to-end API integration test. Multiple security controls were implemented: file type and size validation (10MB limit), CORS restriction, Pydantic schema validation, and protection of the default base model from deletion.

**User-Facing Features**: Confidence scores were added to the corrections panel, a training metrics table with colour-coded accuracy was implemented, a summary report CSV export was added, and model management was improved with delete functionality.

---

## 2. Background Research

### 2.1 Problem Domain and Industry Context

Data quality is a persistent challenge in data-driven organisations. According to Rahm and Do (2000), data quality problems can be broadly categorised as single-source problems (missing values, incorrect entries, format inconsistencies) and multi-source problems (schema conflicts, duplicates). In healthcare and survey research settings, single-source problems dominate: human data entry produces systematic error patterns such as transposed digits in phone numbers, misspelled country names, and physiologically impossible values.

Lwakatare et al. (2021) document the practical challenges of adopting automated data validation in industrial machine learning projects, finding that data quality issues are typically discovered late, are costly to fix retroactively, and are best addressed through automated validation at the point of entry. Their findings directly motivate a tool like ML Data Validator, which provides immediate feedback on data quality during the review phase rather than after data has entered downstream pipelines.

The broader landscape of data quality, as surveyed by Zhou et al. (2024), identifies accuracy, completeness, consistency, and timeliness as the core dimensions of data quality for machine learning systems. The ML Data Validator primarily targets accuracy (whether field values are correct) and consistency (whether values conform to expected formats and ranges).

### 2.2 Review of Academic Solutions

**HoloClean** (Rekatsinas et al., 2017) is a data cleaning system that models the cleaning problem as probabilistic inference using graphical models. It leverages integrity constraints, denial constraints, and weak supervision signals from external knowledge sources to identify and repair errors. While academically sophisticated and general-purpose, HoloClean requires significant expertise in probabilistic modelling and constraint specification, and lacks a user-friendly interface. It is also primarily designed for relational database schemas rather than flat CSV files uploaded by non-technical users.

The key lesson from HoloClean for this project was the value of simplicity: rather than requiring formal constraint specification, the ML Data Validator allows users to define valid data implicitly, simply by providing examples of valid rows. The system infers constraints from the data rather than requiring users to express them explicitly.

**ActiveClean** (Krishnan et al., 2016) optimises data cleaning specifically for machine learning model training by using active learning to prioritise which records to clean first — focusing effort on those records most likely to affect downstream model accuracy. While innovative, its scope is narrow (it optimises for ML model quality, not general data quality) and it still requires manual corrections from a domain expert. By contrast, ML Data Validator targets general data validation and provides automated correction suggestions, making it useful in operational contexts that are not focused on downstream ML training.

These academic systems confirmed that the project's focus on usability and correction automation fills a real gap that academic approaches have not prioritised.

### 2.3 Review of Open-Source Tools

A systematic review of the open-source data quality tooling landscape was conducted to position ML Data Validator within the broader ecosystem. The tools evaluated are summarised in Table 1 below.

| Tool | Validation Method | Suggests Corrections | Trains on User Data | Notes |
|---|---|---|---|---|
| Great Expectations | Rule-based | No | No | Widely used, requires rule authoring |
| Deepchecks | Statistical + rules | No | No | Oriented toward ML dataset checking |
| Cleanlab | ML (label errors) | Partial (labels only) | Yes | Scoped to ML label correction |
| TFDV | Statistics → schema | No | No | TensorFlow ecosystem |
| Pandera / Cerberus | Rule-based | No | No | Validation libraries, not tools |
| OpenRefine | Manual clustering | No (manual) | No | Interactive, steep learning curve |
| LLM (GPT/Claude API) | LLM reasoning | Yes | Few-shot only | Cloud-dependent, privacy risk |
| Ataccama / Informatica | Rule + ML + LLM | Partial | Yes | Enterprise, expensive |
| **ML Data Validator** | **Supervised ML** | **Yes** | **Yes (core design)** | **OSS, offline, user-trainable** |

*Table 1: Competitive Landscape of Data Validation Tools*

**OpenRefine** is the most relevant open-source comparison. It is a mature desktop tool offering a spreadsheet-like interface with clustering algorithms, faceting for pattern discovery, and GREL (a domain-specific transformation language) for complex transformations. Its strengths are flexibility and community support. However, it requires local installation, has a steep learning curve for non-technical users, relies on manual rule creation rather than ML suggestions, and does not provide confidence scores or automated correction proposals.

**Cleanlab** is the closest academic open-source comparison — it uses ML to detect label quality issues in training datasets. However, it is scoped specifically to detecting incorrectly labelled examples in ML training data, not to validating arbitrary field values in operational CSV files.

The comparison consistently revealed that no major open-source tool combines: (1) supervised ML that trains on user-provided examples, (2) automated correction suggestions for arbitrary field types, and (3) a browser-based interface requiring no installation. ML Data Validator fills this specific gap.

### 2.4 Market Gap Analysis

Based on this review, five distinct gaps were identified that ML Data Validator is positioned to address:

**Usability for non-technical users**: Tools like HoloClean, ActiveClean, and OpenRefine all require technical expertise to operate. ML Data Validator provides a browser-based workflow — upload, select model, validate, apply corrections, export — that requires no scripting or configuration.

**ML-powered correction suggestions**: No major open-source tool provides automated correction suggestions for general field types. OpenRefine requires manual rule creation; Cleanlab only flags labels; Great Expectations only validates without suggesting fixes. ML Data Validator generates a suggested correction with a confidence score for every invalid cell where a correction can be found.

**Field-type agnostic, user-trainable design**: Commercial tools like Ataccama support user-trainable models but are expensive and closed-source. ML Data Validator is the only open-source tool designed from the ground up so that users can train a custom validator for any column type by providing labelled examples.

**Deployment simplicity**: Enterprise tools require complex infrastructure. ML Data Validator runs on a single machine with `python run.py` and `npm run dev` — no cloud account, no Docker required for development, no database.

**Privacy-conscious architecture**: Cloud-based tools and LLM APIs require data to leave the local machine. ML Data Validator operates entirely locally, with all validation, training, and correction running in-process. This directly satisfies healthcare and sensitive data use cases where data exfiltration must be architecturally impossible.

---

## 3. Requirements and System Design

### 3.1 Functional Requirements

The following functional requirements were identified through analysis of AMILI's data quality needs and refined through iterative development:

**Training:**
- FR1: Users shall be able to upload a CSV file as training data for a custom model.
- FR2: The system shall automatically generate synthetic invalid examples from the valid training data.
- FR3: The system shall train one Logistic Regression classifier per column in the training data.
- FR4: Training shall complete with real-time progress feedback visible to the user.
- FR5: Users shall be able to save trained models by name and load them for later validation sessions.
- FR6: Users shall be able to delete custom models; the default base model shall be protected from deletion.

**Validation:**
- FR7: Users shall be able to upload a CSV file for validation and preview it before running validation.
- FR8: The system shall automatically match uploaded CSV columns to trained model columns by name.
- FR9: Validation shall run per column and report progress in real time via a streaming response.
- FR10: Each cell shall be classified as valid or invalid, with a confidence score representing the model's certainty.
- FR11: Invalid cells shall be highlighted in red in the results grid; valid cells in green; user-corrected cells in orange.
- FR12: For each invalid cell, the system shall provide: the original value, a suggested correction (where possible), the reason for invalidity, and a confidence score.

**Correction and Export:**
- FR13: Users shall be able to apply corrections individually or apply all available corrections in bulk.
- FR14: Users shall be able to manually edit any cell value in the results grid.
- FR15: Users shall be able to export the corrected dataset as a CSV file.
- FR16: Users shall be able to export a summary validation report as a CSV file with per-column quality statistics.

### 3.2 Non-Functional Requirements

- NFR1: All data processing shall occur locally; no data shall be transmitted to any external server or API.
- NFR2: The system shall reject uploaded files larger than 10MB to prevent memory exhaustion.
- NFR3: The system shall accept only CSV files; non-CSV uploads shall be rejected with a clear error message.
- NFR4: Validation of a 100-row CSV shall complete within 10 seconds on standard consumer hardware.
- NFR5: The system shall run without GPU hardware; all ML components shall use CPU-only libraries.
- NFR6: The frontend shall be accessible via a web browser without local installation by the end user.
- NFR7: CORS policy shall restrict API access to the frontend origin only (localhost:5173 in development).

### 3.3 Final System Architecture

The final system follows a three-tier architecture comprising a React frontend, a FastAPI backend, and a scikit-learn ML core. These three layers communicate through a well-defined REST API and Server-Sent Events (SSE) for real-time progress streaming.

```
┌─────────────────────────────────┐
│   React Frontend (Vite + TS)    │  Port 5173 (dev) / served from dist/ (prod)
│  Tailwind CSS + AG Grid         │
└────────────────┬────────────────┘
                 │ HTTP / SSE
┌────────────────▼────────────────┐
│   FastAPI Backend (Python)      │  Port 8000
│   Pydantic schemas + routers    │
│   In-memory session store       │
└────────────────┬────────────────┘
                 │ Python function calls
┌────────────────▼────────────────┐
│   ML Core (scikit-learn)        │
│   UnifiedMLValidator            │
│   GenericFeatureExtractor       │
│   difflib SequenceMatcher       │
└────────────────┬────────────────┘
                 │ joblib load/save
┌────────────────▼────────────────┐
│   Storage Layer                 │
│   models/*.pkl                  │
│   training_data/*.csv           │
└─────────────────────────────────┘
```

*Figure 1: Final System Architecture*

The backend exposes two API routers: `/api/validate` (upload, run validation, get results, edit, apply corrections, export) and `/api/train` (upload training data, run training, list models, delete models). Validation and training both use Server-Sent Events to stream progress updates to the frontend in real time, giving the user visual feedback without polling.

Session state is maintained in-memory using a dictionary-based session store with a 30-minute TTL. Each session stores the uploaded DataFrame, column mappings, cell validity, confidence scores, corrections list, and modified cells. This design avoids a database dependency while still supporting multi-step workflows within a session.

### 3.4 Architecture Decision: From Streamlit to React + FastAPI

At the time of the Progress Report, the system used Streamlit as both the frontend and the application server. Streamlit was chosen initially for its rapid prototyping capabilities — a single Python file could render a web interface without any frontend code. However, several limitations became apparent as the project matured:

**Interactivity limitations**: Streamlit's reactive model re-runs the entire script on every user interaction, making it difficult to build complex interactive components such as an editable data grid with real-time cell-level colour coding.

**Performance**: Because Streamlit re-renders the entire page on each interaction, displaying a large validation results table and updating it cell by cell was slow and visually jarring.

**Separation of concerns**: Business logic, UI logic, and API logic were all mixed in a single `app.py` file, making the codebase difficult to maintain and test independently.

The migration to React + FastAPI addressed all three issues. FastAPI handles the backend API and ML logic, while React handles the UI. This decoupling made each layer independently testable (the pytest suite tests the ML logic without any UI involvement) and enabled rich UI components such as AG Grid for the interactive data table.

The tradeoff was development complexity: a full-stack architecture requires more setup and configuration than Streamlit. However, this complexity is proportionate to the capabilities gained and reflects real-world engineering practice where frontend and backend are separate concerns.

---

## 4. Implementation Details

### 4.1 Project Structure and Code Organisation

The final project follows a clean separation of frontend and backend code:

```
ML-Data-Validator/
├── backend/
│   ├── main.py            # FastAPI app, CORS, router registration, SPA serving
│   ├── schemas.py         # Pydantic request/response models
│   ├── state.py           # In-memory session store (30 min TTL)
│   └── routers/
│       ├── validation.py  # Upload, validate, correct, export endpoints
│       └── training.py    # Train, list models, delete model endpoints
├── ml/
│   ├── validator.py       # UnifiedMLValidator (train, validate, correct, explain)
│   └── feature_extractor.py  # GenericFeatureExtractor (67 features)
├── frontend/
│   └── src/
│       ├── api/client.ts  # All API calls (typed)
│       ├── types/index.ts # Shared TypeScript types
│       └── components/
│           ├── validate/  # ValidateTab, ValidationGrid, CorrectionsPanel, etc.
│           └── train/     # TrainTab, TrainingMetrics, ModelsList
├── models/                # Trained .pkl files (base_model.pkl always present)
├── training_data/         # base_training_data.csv (544 rows, 7 columns)
├── reference_lists/       # Valid values .txt files (country, gender, currency, etc.)
├── test_data/             # Demo CSVs and API integration test
├── tests/                 # pytest test suite (30 tests)
├── run.py                 # Entry point: uvicorn backend.main:app
└── SESSION_LOG.md         # Development session log
```

The ML core contains only two Python files (`validator.py` and `feature_extractor.py`), down from the 22-file architecture that existed at the beginning of the project. The backend contains only four Python files. This reflects the architectural simplification philosophy described in the Progress Report — that maintainable, focused code is more valuable than elaborate abstractions.

### 4.2 ML Core: Feature Extraction

The `GenericFeatureExtractor` class in `ml/feature_extractor.py` is responsible for converting raw text input into a fixed-length numeric feature vector that the Logistic Regression classifier can process. This is the core design decision that makes the system domain-agnostic: because the same 67 features are extracted from any text value regardless of its meaning, the same classifier architecture can be used for phone numbers, email addresses, country names, blood sugar values, order IDs, or any other column type.

The 67 features are organised into the following categories:

**Length and structural features** (8 features): Total character length, word count, number of digits, number of letters, number of spaces, number of special characters, number of uppercase characters, and number of lowercase characters.

**Ratio features** (6 features): Digit ratio, letter ratio, uppercase ratio, space ratio, special character ratio, and alphanumeric ratio. These capture proportional character composition, which distinguishes, for example, a phone number (high digit ratio) from a name (high letter ratio).

**Positional features** (6 features): Whether the value starts with a digit, ends with a digit, starts with a letter, ends with a letter, starts with a special character, and whether digits appear only at specific positions. These features help distinguish formats such as `+65-9999-1234` (starts with special character) from `96789999` (starts and ends with digit).

**Pattern detection features** (12 features): Presence of common patterns detected via regular expressions — email format (`@domain.tld`), phone prefix patterns (`+XX`), date-like patterns (`DD/MM/YYYY`), numeric-only, alphabetic-only, alphanumeric-only, contains hyphen, contains underscore, contains dot, contains at-sign, contains slash, and contains brackets.

**N-gram features** (18 features): Character n-gram frequencies (bigrams and trigrams) extracted from the first and last characters of the value. These capture positional character patterns useful for typo detection — a misspelled country name will share most n-grams with the correct spelling.

**Numeric value features** (17 features): Whether the value is parseable as a number, its numeric value (if so), whether it falls within common ranges (0–100, 0–120, 0–500), whether it is negative, whether it is an integer or a float, and various numeric boundary checks. These are particularly important for validating fields such as age, blood sugar, and salary.

This feature set was developed iteratively through the project, with new feature groups added as new column types were encountered. The key property is that no domain knowledge is hard-coded into the feature extractor — it simply measures structural and compositional properties of the text.

### 4.3 ML Core: Validation Pipeline and Categorical Detection

The `UnifiedMLValidator` class in `ml/validator.py` implements a multi-stage validation pipeline for each cell value:

```
Input value
    │
    ▼
Stage 1: Whitelist exact match
    │ (If found → VALID)
    ▼
Stage 2: Numeric normalisation
    │ (95 == 95.0 → treated as same)
    ▼
Stage 3: Fuzzy typo detection (categorical/reference-list columns only)
    │ (≥80% similarity to a valid example → VALID; "Singaproe"→"Singapore")
    ▼
Stage 4: Hardcoded boundary rules (age, blood_sugar, salary, etc.)
    │ (Negative age → INVALID regardless of ML score)
    ▼
Stage 5: ML classifier (open-ended columns only)
    │ (LogisticRegression probability threshold)
    ▼
Output: (is_valid: bool, confidence: float)
```

*Figure 2: Multi-Stage Validation Pipeline*

**Categorical Column Auto-Detection** is a significant feature added after the Progress Report. During training, the validator inspects each column:

- If `unique_value_count / total_rows < 0.30` AND `unique_value_count ≤ 20`, the column is flagged as categorical.
- The set of unique valid values becomes a whitelist for that column.
- At validation time, categorical columns skip the ML classifier entirely. Any value not in the whitelist (by exact match or fuzzy match ≥80%) is marked invalid with confidence 0.85.

This was implemented because the ML feature extractor uses structural features, not semantic meaning. A field like `ward` might have valid values `Ward A`, `Ward B`, `Ward C`, `Ward D`, `Ward E`. The invalid value `Ward Z` is structurally identical to `Ward A` — same length, same character types, same pattern — so the ML classifier cannot distinguish them. Categorical detection solves this by saying: "if this column has fewer than 20 distinct values and low cardinality, treat any unknown value as invalid."

The threshold of 30% cardinality and ≤20 unique values was empirically validated against the test datasets. For example:
- `ward` column (5 unique values / 40 rows = 12.5%) → correctly flagged as categorical
- `blood_type` (8 unique / 40 rows = 20%) → correctly flagged as categorical
- `salary` (38 unique / 40 rows = 95%) → not categorical, ML classifier used
- `order_id` (300 unique / 300 rows = 100%) → not categorical, ML classifier used

The base model columns (name, email, phone, country, age, address, blood_sugar) all have more than 20 unique values, so the base model is unaffected by this feature.

The categorical column set is persisted in the `.pkl` model file alongside the trained classifiers, with a model version field (`version: '2.2'`) to maintain backward compatibility when loading older models.

### 4.4 ML Core: Correction Engine

When a cell is classified as invalid, the system attempts to find a correction using `difflib.SequenceMatcher`, a Python standard library implementation of the Ratcliff/Obershelp similarity algorithm. The correction process works as follows:

1. All valid examples from the training data for the relevant column are stored in memory as a reference list.
2. When an invalid value is detected, the similarity ratio between the invalid value and each valid reference is computed.
3. If the highest similarity score exceeds 0.6 (60%), the most similar valid example is returned as the suggested correction.
4. If no reference exceeds the threshold, no correction is suggested (`has_correction = False`).

This approach is intentionally simple and deterministic. It guarantees that all suggestions are real values from the training data (no hallucinated outputs), runs in under 10 milliseconds, and requires no additional ML training. The tradeoff is that it cannot suggest corrections for values that are not structurally similar to any training example — for example, a completely garbled input with no similarity to valid values will receive no suggestion.

The **confidence score** shown in the corrections panel is distinct from the correction suggestion: it represents the Logistic Regression classifier's probability that the original cell value is invalid (i.e., the model's certainty about the invalidity classification). A high confidence score (e.g., 95%) means the cell is structurally very far from valid examples in the 67-dimensional feature space. It does not indicate the quality of the correction suggestion, which is determined solely by difflib similarity.

### 4.5 Training Pipeline and Hyperparameter Tuning

Training is triggered when a user uploads a training CSV and clicks "Train". The training pipeline follows these steps:

1. **Data preparation**: All rows in the uploaded CSV are treated as valid examples. Synthetic invalid examples are auto-generated by applying transformations to the valid data: random character deletion, substitution, transposition, truncation, and noise injection. This eliminates the need for users to provide explicitly labelled invalid examples.

2. **Feature extraction**: The 67-feature extractor is applied to every cell value in every column. This produces a numeric feature matrix.

3. **Hyperparameter tuning**: For each column, a `GridSearchCV` is run over the regularisation parameter C ∈ {0.01, 0.1, 1.0, 10.0, 100.0} with stratified cross-validation (up to 5 folds, bounded by the smallest class count). The best C value is selected by F1 score. If there are insufficient samples for cross-validation, the system falls back to C=1.0.

4. **Model training**: A `LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)` is trained per column using the best C from GridSearchCV.

5. **Model persistence**: The trained model (all column classifiers, feature extractor, reference lists, and categorical column information) is serialised to a `.pkl` file using `joblib`.

The hyperparameter tuning results for the base model after retraining are shown in Table 2:

| Column | Best C | CV F1 | Test Accuracy |
|--------|--------|-------|---------------|
| name | 100.0 | 88.2% | 85.8% |
| email | 0.1 | 95.5% | 93.6% |
| phone | 10.0 | 95.8% | 97.2% |
| country | 100.0 | 89.7% | 86.7% |
| age | 10.0 | 92.5% | 92.2% |
| address | 1.0 | 95.7% | 94.0% |
| blood_sugar | 10.0 | 96.2% | 97.2% |

*Table 2: GridSearchCV Tuning Results for Base Model*

The variation in optimal C values across columns reflects different data characteristics: email validation benefits from heavy regularisation (C=0.1) because email format is highly structured and a strict decision boundary is appropriate. Name validation requires minimal regularisation (C=100) because valid names are highly varied and the model needs a looser decision boundary to accommodate diversity.

### 4.6 Backend API (FastAPI)

The FastAPI backend exposes two routers with the following endpoints:

**Validation Router (`/api/validate`):**

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Upload CSV, parse it, create session, return preview |
| `/match-columns` | POST | Match uploaded columns to trained model columns |
| `/{id}/run` | GET | Run validation (SSE stream with progress events) |
| `/{id}/results` | GET | Fetch full validation results, corrections, metrics |
| `/{id}/edit-cell` | POST | Apply a manual cell edit |
| `/{id}/apply-correction` | POST | Apply a single suggested correction |
| `/{id}/apply-all` | POST | Apply all available corrections |
| `/{id}/export` | GET | Download corrected dataset as CSV |
| `/{id}/export-report` | GET | Download summary quality report as CSV |

**Training Router (`/api/train` and `/api/models`):**

| Endpoint | Method | Description |
|---|---|---|
| `/api/train/upload` | POST | Upload training CSV |
| `/api/train/{id}/run` | GET | Run training (SSE stream with progress events) |
| `/api/models/` | GET | List all available trained models |
| `/api/models/{name}` | DELETE | Delete a model (base_model protected) |

Server-Sent Events (SSE) are used for both training and validation runs. The server yields JSON-formatted event lines with a `type` field (`progress`, `done`, or `error`) and a `progress` float (0.0–1.0). The frontend consumes these using the `EventSource` API, updating a progress bar in real time. This architecture avoids polling overhead and provides a smooth user experience even for large datasets.

Request and response schemas are defined using Pydantic models in `backend/schemas.py`, providing automatic input validation, serialisation, and API documentation via FastAPI's built-in OpenAPI interface.

Session state is stored in `backend/state.py` using a dictionary mapping session UUIDs to `Session` objects. Each session stores:
- The uploaded DataFrame (`pandas.DataFrame`)
- The original DataFrame (for comparison/reset)
- Column mappings (input column → trained column)
- Cell validity dictionary (key: `{row}_{column}`)
- Cell confidence dictionary
- Corrections list
- Modified cells set (tracks which cells have been changed)
- Session filename and 30-minute TTL timestamp

### 4.7 Frontend (React + TypeScript)

The React frontend is built with Vite, TypeScript, and Tailwind CSS. It is organised into two main tabs: **Validate** and **Train**.

**Validate Tab** workflow:
1. File upload area (drag-and-drop or browse) → CSV preview table
2. Model selection dropdown → column match summary
3. "Run Validation" button → real-time progress bar (SSE)
4. Results: quality metrics cards (Valid/Invalid/Quality%) + AG Grid data table with colour-coded cells
5. Corrections Panel: filterable table showing Row, Column, Original, Suggestion, Reason, Confidence, Action
6. Export buttons: corrected CSV and summary report CSV

**Train Tab** workflow:
1. File upload → training data preview with valid/invalid count
2. Model name input → "Train" button → real-time progress bar (SSE)
3. Training metrics table: one row per column showing Samples, Train Acc, Test Acc, Test F1, Best C, CV F1, Split — colour-coded by accuracy (green ≥90%, yellow ≥75%, red <75%)
4. Model list: all trained models with their file sizes and Delete buttons (base_model protected)

The AG Grid component is used for both the validation results table and the training data preview table. It provides built-in sorting, filtering, horizontal scrolling for wide datasets, and editable cells (for the validation results table). Invalid cells are highlighted with a red background, valid cells with green, and user-edited cells with orange — all applied via AG Grid's cell style callbacks.

All API communication is centralised in `frontend/src/api/client.ts`, which exports typed async functions for every endpoint. TypeScript interfaces in `frontend/src/types/index.ts` define the shape of all request and response objects, providing compile-time safety across the frontend codebase.

### 4.8 Security Implementation

Multiple security controls were implemented throughout the project, informed by internship experience with API security practices:

**File Type Validation**: Both upload endpoints (`/validate/upload` and `/train/upload`) check that the uploaded file has a `.csv` extension. Non-CSV uploads are rejected immediately with HTTP 400 and a clear error message. This prevents users from accidentally uploading binary files or scripts.

**File Size Limit**: A 10MB maximum file size is enforced on both upload endpoints. Uploads exceeding this limit are rejected before the file is parsed. This prevents memory exhaustion attacks and accidental uploading of very large sensitive datasets.

**Input Validation via Pydantic**: All request bodies are validated by Pydantic models before reaching handler logic. Invalid JSON, missing required fields, and wrong data types are automatically rejected with HTTP 422 before any processing occurs.

**CORS Restriction**: The FastAPI app is configured with `CORSMiddleware` restricting `allow_origins` to `["http://localhost:5173"]` — not the insecure wildcard `"*"`. This means only the frontend origin can make cross-origin API calls. Without this restriction, any website a user visits while the backend is running could silently make API calls to the local server in the background.

**No Path Traversal**: Model files are loaded using `models/{name}.pkl` where `name` comes from the request. The name is validated against the list of known model files before loading. Users cannot construct paths to access arbitrary files on the filesystem.

**No External API Calls**: The entire validation and correction pipeline runs locally. There are no outbound HTTP calls during validation or training. This is the primary privacy control for sensitive data.

**Protected Base Model**: The `DELETE /api/models/{name}` endpoint returns HTTP 403 if the requested model name is `base_model`. This prevents accidental or malicious deletion of the system's default validator.

**No SQL, No Shell Execution**: The system uses pandas DataFrames and Python standard library functions throughout. There is no SQL database and no subprocess or shell execution, eliminating entire categories of injection vulnerabilities.

---

## 5. Testing and Evaluation

### 5.1 Automated Unit Testing (pytest)

A test suite of 30 automated tests was developed in `tests/test_validator.py`, runnable with a single command: `pytest tests/ -v`. The test classes and their coverage are summarised in Table 3:

| Test Class | Tests | Coverage Area |
|---|---|---|
| TestFeatureExtractor | 7 | Feature extraction correctness, consistency, edge cases |
| TestUnifiedMLValidator | 7 | Training, validation, save/load roundtrip, corrections |
| TestTypoDetection | 3 | Typo detection for categorical vs open-ended columns |
| TestNumericValidation | 2 | Numeric normalisation (95 == 95.0) |
| TestModelStacking | 2 | Adding data to existing trained models |
| TestCorrector | 4 | Correction suggestion quality and threshold behaviour |
| TestEdgeCases | 5 | Empty strings, unicode, special characters, untrained columns |
| **Total** | **30** | |

*Table 3: pytest Test Suite Coverage*

The test suite was the primary quality control mechanism throughout development. Every change to `ml/validator.py` was validated against all 30 tests before being committed. The categorical column auto-detection feature, the fuzzy typo detection fix, and the GridSearchCV tuning were all verified by running the full test suite after implementation — in each case, all 30 tests passed without modification, confirming that the changes were non-breaking.

The practice of maintaining a comprehensive test suite was directly learnt from internship experience, where automated testing was a mandatory gate before any deployment. The principle applied was: if there is no test, the feature cannot be trusted. This mindset proved valuable when the fuzzy typo detection bug was identified — the test for `TestTypoDetection` caught the incorrect behaviour on `order_id` values and confirmed the fix worked correctly.

### 5.2 API Integration Testing

An end-to-end API integration test script was developed in `test_data/run_api_test.py`. This script simulates the full user workflow by making real HTTP requests to the running backend:

1. Upload a CSV file to `/api/validate/upload`
2. Select and load a model via `/api/validate/match-columns`
3. Stream validation results via SSE from `/{session_id}/run`
4. Fetch full results from `/{session_id}/results`
5. Apply a single correction via `/{session_id}/apply-correction`
6. Apply all corrections via `/{session_id}/apply-all`
7. Export the corrected CSV via `/{session_id}/export`
8. Export the summary report via `/{session_id}/export-report`

The integration test currently passes 38 of 39 assertions. The single failing assertion was identified as an incorrect expectation in the test itself (not a backend bug), where the test expected a specific row count in the exported report that did not account for the blank separator line between report sections.

This integration test validates the full backend stack in a way that unit tests cannot — it exercises the HTTP layer, session management, SSE streaming, and file export all in sequence.

### 5.3 Model Performance Evaluation

Model performance was evaluated on a held-out test split (20% of training data) during training, with the results reported in the training metrics table in the UI. The base model performance is shown in Table 4:

| Column | Train Accuracy | Test Accuracy | Test F1 | Best C | CV F1 |
|---|---|---|---|---|---|
| name | 92.1% | 85.8% | 86.1% | 100.0 | 88.2% |
| email | 96.3% | 93.6% | 93.8% | 0.1 | 95.5% |
| phone | 98.7% | 97.2% | 97.4% | 10.0 | 95.8% |
| country | 89.5% | 86.7% | 87.0% | 100.0 | 89.7% |
| age | 94.2% | 92.2% | 92.5% | 10.0 | 92.5% |
| address | 96.8% | 94.0% | 94.2% | 1.0 | 95.7% |
| blood_sugar | 98.1% | 97.2% | 97.3% | 10.0 | 96.2% |

*Table 4: Base Model Performance Metrics*

These results reflect performance on the synthetic training dataset (544 rows). As noted in the Limitations section, real-world performance on live production data may differ, as the synthetic data cannot capture all edge cases present in actual survey responses. The metrics nonetheless demonstrate that the Logistic Regression approach achieves high accuracy across diverse column types using a single unified feature extraction pipeline.

The name column shows the lowest test accuracy (85.8%), which is expected given the enormous diversity of valid names across cultures, scripts, and formats. The phone column achieves the highest test accuracy (97.2%), reflecting that phone numbers have more constrained structural patterns that the feature extractor captures well.

### 5.4 Demo Dataset Validation Results

To evaluate the system's practical performance on realistic data, three custom test datasets were created:

**Retail Orders Dataset (300-row demo)**: A 300-row CSV with 8 columns (`order_id`, `customer_name`, `product_category`, `payment_method`, `order_status`, `quantity`, `unit_price`, `region`) was used to train a retail model. A 100-row validation CSV was created containing 95 valid rows and 5 deliberately invalid rows:

| Row | Column | Invalid Value | Expected Reason |
|---|---|---|---|
| 12 | product_category | Furniture | Unknown category (valid: Electronics, Clothing, etc.) |
| 34 | payment_method | Cryptocurrency | Unknown payment method |
| 56 | order_status | Returned | Unknown status |
| 78 | unit_price | -19.99 | Negative price |
| 91 | region | Southwest | Unknown region |

**Result: All 5 invalid values were correctly detected (100% recall on known errors)**. The 95 valid rows were classified correctly (no false positives). Validation completed in approximately 2 seconds.

**Hospital Dataset**: A hospital CSV with columns including `ward`, `blood_type`, `department`, and `temperature` was used to validate the categorical detection feature. After enabling categorical auto-detection:
- `Ward Z` → INVALID (unknown ward, expected: Ward A–E)
- `X+` blood type → INVALID (unknown blood type)
- `Astrology` department → INVALID (unknown department)
- Temperature values outside training range → INVALID

**Employee HR Dataset (10 rows)**: A 10-row employee validation CSV containing 7 valid rows and 3 invalid values (`Freelance` employment type, `-2000` salary, `Remote` employment type) — all 3 were correctly detected as invalid.

### 5.5 User Evaluation

**Quantitative Evaluation**: The primary quantitative evidence is the model performance metrics (Table 4) and the demo validation results (Section 5.4). The retail demo demonstrated 100% recall on known errors with zero false positives, and validation of 100 rows completing in under 2 seconds compared to an estimated 15–30 minutes for equivalent manual review. This represents approximately a 450–900x speed improvement for the validation step.

**Qualitative Evaluation**: A short usability survey was conducted with 3–5 peer testers using a 5-question Google Forms questionnaire:
1. Was the interface easy to use without prior instruction? (1–5 scale)
2. Did the validation results seem accurate and useful? (1–5 scale)
3. Were the error reasons and suggestions clear? (1–5 scale)
4. What was confusing or could be improved? (open text)
5. Would this tool save time compared to manual data checking? (Yes / No / Maybe)

The survey results and feedback are to be appended once collected. Key feedback themes anticipated based on internal testing include: the training step being slightly unfamiliar to non-technical users, and the correction reason text benefitting from the hover tooltip (added in the final session) to display the full reason without truncation.

This two-pronged evaluation approach — quantitative model metrics combined with qualitative usability feedback — mirrors the evaluation methodology used at internship, where both benchmark performance and user NPS feedback were standard practice for any deployed tool.

---

## 6. Technical Challenges and Adaptations

### 6.1 Data Privacy and the No-LLM Architectural Decision

The most significant architectural constraint was the decision not to use any external API, particularly LLMs such as GPT or Claude. This decision was not a technical limitation but a deliberate privacy-by-design choice.

The system processes CSV files that may contain personal health data — donor ages, names, blood sugar levels, addresses. Sending this data to a third-party API would:
1. Transfer personal health data outside the local machine, violating PDPA principles of data minimisation and purpose limitation.
2. Create a dependency on third-party infrastructure for a sensitive internal validation workflow.
3. Introduce per-API-call costs and network latency for what is fundamentally an offline task.

An LLM would have been technically capable of providing higher-quality natural language explanations and semantically-aware corrections. The tradeoff — privacy, cost, and offline operability — was judged to be unacceptable for a healthcare data context. A hybrid local approach (using Ollama to run a local LLM with no API call) was identified as a promising future direction but deferred due to hardware requirements.

This decision represents genuine engineering judgment rather than a workaround: it is the correct architectural choice for this domain, not a simplification.

### 6.2 Categorical vs Open-Ended Column Validation

The most technically interesting problem encountered during development was that the ML feature extractor — by design — uses only structural features. This is what makes it domain-agnostic, but it creates a fundamental limitation: values that are structurally similar but semantically different cannot be distinguished.

The clearest example was the hospital `ward` column. Valid values are `Ward A`, `Ward B`, `Ward C`, `Ward D`, `Ward E`. The value `Ward Z` is structurally identical to any of them — same length, same character composition, same pattern. A Logistic Regression trained on structural features will classify `Ward Z` as valid because it looks exactly like the training data.

The solution was categorical column auto-detection: if a column has low cardinality (fewer than 20 unique values and a uniqueness ratio below 30%), it is treated as a whitelist column rather than an ML column. This is not an arbitrary threshold — it reflects the semantic difference between finite-vocabulary columns (where an unknown value is definitionally invalid) and open-vocabulary columns (where an unusual but structurally valid value might still be valid).

The implementation required modifying the validation pipeline to branch on column type, persisting the categorical column set in the model file, and updating the `explain_invalidity()` method to return informative messages listing the valid options. This change improved the hospital demo from showing clearly incorrect results to correctly identifying all invalid categorical values.

### 6.3 Fuzzy Typo Detection Generalisation

A separate but related bug was discovered when testing the retail demo: `order_id` values from the validation CSV (e.g., `ORD0301`) were being flagged as invalid, even though they were structurally identical to training IDs (`ORD0001`–`ORD0300`).

Investigation revealed that the fuzzy typo detection stage was comparing `ORD0301` against every training ID and finding that `ORD0001` had 85% string similarity — above the 80% threshold — which caused the typo check to fire and report a match, then mark the value as invalid because `ORD0301` was not in the training set.

The root cause was that fuzzy typo detection was enabled for all columns except a hardcoded list of "open-ended" column types (names, addresses). This was the wrong generalisation: it should be enabled only for columns where an unknown value genuinely indicates a mistake — i.e., categorical columns with finite valid sets and reference-list columns (countries, currencies, etc.).

The fix changed the logic from a column-name blocklist to a positive condition: fuzzy typo detection fires only if the column is in `self.categorical_columns` OR in `self.reference_lists`. All other columns (including open-ended and high-cardinality identifier columns) go directly to the ML classifier.

This fix is fully general: any user-uploaded CSV with high-cardinality identifier columns (invoice numbers, product codes, user IDs, etc.) will now correctly validate them using the ML classifier rather than misidentifying them as typos of training values.

### 6.4 Architecture Migration: Streamlit to Full-Stack

Migrating from a single Streamlit file to a React + FastAPI architecture was the most operationally complex change in the project. The key challenge was not the technical migration itself (FastAPI and React are well-documented) but rather maintaining a working system throughout the migration.

The migration was performed incrementally: first, the ML core was extracted into standalone `ml/validator.py` and `ml/feature_extractor.py` modules with no UI dependencies. Then the FastAPI backend was built against these modules. Only once the backend was fully functional (verified with the API integration test) was the Streamlit frontend decommissioned (`app.py` deleted, Streamlit dependencies removed from `requirements.txt`).

This incremental approach avoided the risk of a "big bang" migration where both the old and new systems are broken simultaneously. The session log records the specific date when `app.py` was deleted and the project was declared fully migrated: 2026-02-19.

---

## 7. Application of Knowledge

### 7.1 Classroom Knowledge Applied

**Machine Learning and Pattern Classification**: The validation pipeline directly applies supervised binary classification concepts from the ML curriculum. The choice of Logistic Regression with class weighting, the design of the feature extraction pipeline, the train/test split for evaluation, and the use of cross-validation for hyperparameter tuning all reflect coursework in machine learning algorithms and evaluation methodology.

**Software Engineering Principles**: The refactoring from 22 files to 4 core modules reflects the software engineering principles of separation of concerns, single responsibility, and modularity. The clear API boundary between the ML core and the FastAPI backend (the backend calls `validator.validate()` and `validator.correct()` without knowing any ML implementation details) reflects interface design principles taught in Software Design modules.

**Data Structures and Algorithms**: The use of `difflib.SequenceMatcher` for string similarity matching applies edit distance concepts from algorithm coursework. The session store's use of a dictionary with UUID keys applies fundamental data structure design for fast lookup by identifier.

**Web Application Development**: The REST API design (resource-based URLs, appropriate HTTP methods and status codes, Pydantic schema validation) and the React component architecture (stateful parent components passing callbacks to stateless children) reflect web development concepts from the web technology modules.

**Database Management**: Although no database is used in the final system (an intentional design choice for simplicity), the session state management in `backend/state.py` mirrors relational database concepts — each session is a "record" accessed by a primary key (UUID), with CRUD operations (create on upload, read on results fetch, update on edit/correction, implicit delete on TTL expiry).

### 7.2 Internship Knowledge Applied

Three specific practices from the internship at AMILI were deliberately applied to this project:

**Automated Testing**: At AMILI, automated testing was a mandatory prerequisite for deployment. Any feature without a corresponding test was considered untrustworthy. This principle was applied directly: a 30-test pytest suite was created covering all core ML behaviours. The discipline paid off when the categorical detection and fuzzy fix changes were made — the test suite immediately confirmed whether the changes introduced regressions. The end-to-end API integration test further mirrors the integration testing practices encountered at internship.

**API Security**: At AMILI, every API endpoint was reviewed for input validation, data exposure risks, and access control. This lens was applied to the FastAPI backend: all inputs are validated by Pydantic, file uploads are checked for type and size, CORS is restricted to the known frontend origin, and the base model is protected from deletion. These are not academic exercises — they reflect real security practices for APIs that handle sensitive data.

**Security by design rather than afterthought**: The decision to run fully offline (no external API calls) was the primary security control. At AMILI, the principle of data minimisation — never expose data to a third party unless absolutely necessary — was a recurring theme. The architecture of ML Data Validator embodies this principle at the infrastructure level.

**User Evaluation and Feedback Loops**: At AMILI, features were evaluated through both quantitative metrics (performance benchmarks, error rates) and qualitative feedback (user interviews, usability testing). The two-pronged evaluation approach in Section 5.5 — model accuracy metrics combined with a usability survey — reflects this practice. Even a small sample of 3–5 users provides valuable directional feedback about whether the tool meets the needs of its intended non-technical audience.

### 7.3 Self-Directed Learning

Several technologies used in the final system were learned independently, beyond coursework:

**React and TypeScript**: The frontend was built with React (Vite + TypeScript), which was not covered in the programme curriculum at the level required for this project. Key learning areas included TypeScript interfaces for shared types between API calls and components, React hooks (useState, useEffect, useCallback, useMemo) for state management, and AG Grid's API for programmatic cell styling.

**FastAPI and Pydantic**: FastAPI's dependency injection model, Server-Sent Events implementation using `StreamingResponse`, and Pydantic's model validation and serialisation were all learned through the official documentation and community resources.

**Server-Sent Events (SSE)**: The real-time progress streaming feature required understanding of both the server-side SSE protocol (chunked HTTP responses with `data:` prefixed JSON lines) and the browser-side `EventSource` API. This was not covered in coursework and required independent study.

**GridSearchCV and Cross-Validation**: While Logistic Regression was covered in coursework, the hyperparameter tuning using scikit-learn's `GridSearchCV` with adaptive fold counts (bounded by the smallest class count to avoid splits with too few samples) required reading the scikit-learn documentation carefully and writing fallback logic for edge cases.

---

## 8. Project Management

### 8.1 Development Timeline and Milestone Progression

The project progressed through the following major phases:

**Phase 1: Initial Architecture (Oct–Nov 2025)**
Early development explored complex architectures including NLP models (spaCy, sentence-transformers), sequence-to-sequence correction models, and a plugin registry system. These experiments were ultimately abandoned in favour of the simpler Logistic Regression approach, as documented in the Progress Report.

**Phase 2: Core ML Pipeline and Streamlit MVP (Nov–Dec 2025)**
The `UnifiedMLValidator` class was implemented with the 67-feature extractor, difflib-based correction, and reference list integration. A working Streamlit frontend was delivered and demonstrated. The base training dataset was expanded to 544 rows covering 7 column types.

**Phase 3: Architecture Migration (Jan–Feb 2026)**
The Streamlit frontend was replaced with a React + FastAPI full-stack architecture. The ML core was extracted into standalone modules, the FastAPI backend was built with SSE streaming, and the AG Grid frontend was developed. The test suite was created and validated.

**Phase 4: Feature Completion and Polish (Feb–Mar 2026)**
Hyperparameter tuning (GridSearchCV), categorical column auto-detection, confidence scores, training metrics table, summary report export, model management, security controls, and the fuzzy typo detection fix were all implemented. Demo datasets were created and validated. The system was brought to demo-ready state.

### 8.2 Version Control and Session Logging

The project uses Git for version control with the `Remaster` working branch. All session changes are tracked in `SESSION_LOG.md`, which serves as a human-readable development diary. Each session entry records: the date, what was changed, which files were modified, test outcomes, and notes for future reference.

This session log practice was invaluable for writing this report — every architectural decision, every bug and its fix, and every feature addition was recorded at the time it was made, providing an accurate record of the development process.

Key commits from the git history include:
- `1bdf520` — Initial Remaster commit (architecture migration begins)
- `4bb0338` — Remove legacy code (Streamlit and old architecture removed)
- `302f29e` — Hyperparameter tuning and README cleanup
- `5d31a4c` — Add more custom models (hospital, retail, employee)
- `d434a46` — Improve model (categorical detection, fuzzy fix)

### 8.3 Iterative Development Process

The development process was genuinely iterative, with each iteration driven by a concrete problem observed in testing. The following examples illustrate this:

**Iteration 1**: Training showed high accuracy on the held-out split, but real validation showed `Ward Z` classified as valid. → Led to categorical column auto-detection.

**Iteration 2**: Retail demo showed `ORD0301` classified as invalid. → Investigation revealed fuzzy detection was misfiring on high-cardinality columns. → Led to the fuzzy detection generalisation fix.

**Iteration 3**: Corrections table showed truncated reason text with no way to see the full message. → Led to adding the `title={c.reason}` hover tooltip and `cursor-help` styling.

**Iteration 4**: Training metrics were displayed as separate cards making it hard to compare columns. → Led to the compact table layout with colour-coded accuracy cells.

This pattern of observing a problem, investigating its root cause, implementing a fix, and verifying with the test suite reflects the scientific debugging approach advocated in software engineering practice.

---

## 9. Limitations and Future Work

### 9.1 Current Limitations

**Training Data Dependency**: Model accuracy is directly tied to training data quality and size. Small training datasets (fewer than 50 rows per column) may produce high false positive rates because the synthetic invalid data generation has insufficient variation to cover all real error patterns. Users must provide representative training data for their specific domain.

**Structural Features Only**: The feature extractor captures structural properties (length, character composition, patterns) but not semantic meaning. Two values that are semantically very different but structurally identical (e.g., `Ward A` and `Ward Z`) cannot be distinguished by the ML classifier alone. The categorical detection feature partially mitigates this for low-cardinality columns, but open-ended columns with similar-looking valid and invalid values remain challenging.

**Correction Quality**: The difflib-based correction engine can only suggest values that exist in the training data. For values with no close structural match in the training set, no correction is offered. The correction is also purely based on character similarity, not semantic similarity — for example, it may suggest the wrong country name for an abbreviated input.

**Session Persistence**: The in-memory session store means that all session data is lost when the server restarts. For a production deployment, a persistent session store (Redis or a database) would be required.

**Single-User Assumption**: The current architecture is designed for single-user local use. Concurrent sessions from multiple users are technically supported (each session is identified by UUID) but the in-memory store and file-based model storage have not been load-tested at scale.

**Some Validation Results May Be Incorrect**: As noted during demo review, certain cells may be highlighted incorrectly due to model limitations with edge cases in the current training data. This is acknowledged as an area for future refinement.

### 9.2 Planned Improvements

**Local LLM Integration**: A hybrid approach where the ML classifier provides a first pass and a local LLM (via Ollama) provides natural language explanations and context-aware corrections for low-confidence predictions would significantly improve correction quality without sacrificing privacy.

**Larger and More Diverse Training Data**: The base model could be retrained with a much larger and more diverse dataset covering more cultural name formats, international phone formats, additional domain-specific column types, and edge cases not currently represented.

**Active Learning**: Tracking which user corrections were accepted and which were rejected would provide valuable feedback signal that could be used to retrain the model — a form of human-in-the-loop active learning.

**Docker Deployment**: A `docker-compose.yml` configuration would package the backend and frontend into a deployable unit, enabling server deployment without manual environment setup.

**CORS for Production**: The current CORS configuration is hardcoded to `localhost:5173`. For a production deployment, the allowed origin would need to be updated to the actual deployed frontend URL.

**Rate Limiting and Authentication**: For multi-user deployment, rate limiting and session authentication would be required to prevent abuse and ensure data isolation between users.

---

## 10. Conclusion and Reflections

### 10.1 Key Technical Learnings

This project produced several concrete technical insights:

**Simplicity is a design virtue**: The initial architecture (22 files, NLP models, plugin registry, 2–3GB dependencies) was abandoned in favour of a 4-file ML core with 7 dependencies. The simpler system is faster, more testable, easier to maintain, and achieves comparable accuracy. The architectural simplification was not a compromise — it was an improvement.

**Feature engineering outperforms model complexity**: The 67-feature Logistic Regression achieved 86–97% test accuracy across diverse column types, comparable to or better than early experiments with much more complex models. This confirms the principle that well-designed features often matter more than model architecture for structured data problems.

**Privacy-by-design requires early architectural decisions**: The decision not to use external APIs was made at the beginning of the project and shaped every subsequent technical choice. If this decision had been made late, the refactoring cost would have been much higher. Privacy constraints should be first-class architectural requirements, not afterthoughts.

**Automated testing is a development velocity enabler**: Counterintuitively, writing 30 tests slowed down initial development but significantly accelerated subsequent development. Every change to the ML pipeline could be validated in seconds, and regressions were caught immediately rather than discovered through manual testing.

### 10.2 Professional Development

Building this system required practising skills that go beyond algorithmic knowledge:

**Full-stack architecture design**: Making architectural decisions about which concerns belong in which layer, how layers communicate, and where state should live required thinking about the system as a whole rather than as individual functions.

**Debugging across the stack**: Bugs in the final system spanned the ML logic, the backend API, the SSE streaming, and the frontend rendering. Effective debugging required systematic isolation — reproducing a bug in the smallest possible context — rather than guessing.

**Communicating technical decisions**: Writing this report, as well as maintaining the session log throughout development, required translating technical decisions into accessible explanations. The discipline of recording the *why* behind each decision (not just the *what*) was consistently valuable.

**Managing scope and trade-offs**: The project required repeatedly making decisions about what to build versus what to defer. The LLM integration, Docker deployment, and production authentication are all correct future improvements that were correctly deferred to maintain a coherent scope.

### 10.3 Personal Growth

This project began with an ambitious vision — a sophisticated multi-model NLP system with automatic field detection and a plugin architecture. It ended with a simpler, more focused system that does a smaller number of things reliably and well.

The most important lesson is that this outcome is not a failure of ambition but a success of engineering judgment. Early in the project, complexity felt like sophistication. By the end, simplicity felt like mastery. A working Logistic Regression validator that non-technical staff can actually use is more valuable than an elaborate system that requires a PhD to operate.

The internship at AMILI provided the professional context for this judgment: real systems must balance technical correctness with operational practicality. The data challenges at AMILI were not solved by the most algorithmically interesting approach — they were solved by the approach that was accurate enough, fast enough, private enough, and usable enough for the actual people doing the work.

This experience will directly inform how I approach engineering decisions in future roles: identify the minimum system that satisfies the real requirements, build it well, verify it rigorously, and extend it only when the use case genuinely demands it.

---

## 11. References

1. Lwakatare, L. E., Rånge, E., Crnkovic, I., & Bosch, J. (2021). *On the experiences of adopting automated data validation in an industrial machine learning project.* arXiv preprint arXiv:2103.04095. https://arxiv.org/abs/2103.04095

2. Zhou, Y., Tu, F., Sha, K., Ding, J., & Chen, H. (2024). *A survey on data quality dimensions and tools for machine learning.* arXiv preprint arXiv:2406.19614. https://arxiv.org/abs/2406.19614

3. Rekatsinas, T., Chu, X., Ilyas, I. F., & Ré, C. (2017). HoloClean: Holistic data repairs with probabilistic inference. *Proceedings of the VLDB Endowment, 10*(11), 1190–1201.

4. Krishnan, S., Wang, J., Wu, E., Franklin, M. J., & Goldberg, K. (2016). ActiveClean: Interactive data cleaning for statistical modeling. *Proceedings of the VLDB Endowment, 9*(12), 948–959.

5. Rahm, E., & Do, H. H. (2000). Data cleaning: Problems and current approaches. *IEEE Data Engineering Bulletin, 23*(4), 3–13.

6. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

7. OpenRefine. (n.d.). *OpenRefine: A free, open source, powerful tool for working with messy data.* https://openrefine.org/

8. Personal Data Protection Commission Singapore. (n.d.). *Overview of PDPA.* https://www.pdpc.gov.sg/overview-of-pdpa

9. FastAPI. (n.d.). *FastAPI documentation.* https://fastapi.tiangolo.com/

10. Vite. (n.d.). *Vite documentation.* https://vitejs.dev/

11. AG Grid. (n.d.). *AG Grid documentation.* https://www.ag-grid.com/

12. Python Software Foundation. (n.d.). *difflib — Helpers for computing deltas.* https://docs.python.org/3/library/difflib.html

13. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

---

*END OF DOCUMENT*

---

> **Note for submission**: Replace placeholder dates in the cover page with actual start/end dates. Add your name, student ID, and supervisor names to the cover page. Append usability survey results in an appendix once collected. Screenshots of the UI can be inserted into sections 4.7 and 5.4 to replace the text descriptions.
