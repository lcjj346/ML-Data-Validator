# Claude Code Instructions

## Project Overview
ML Data Validator - A Streamlit app for training ML models to validate data (phone numbers, emails, names, etc.) and suggest corrections.

- **Run:** `streamlit run app.py`
- **Main branch:** `main`
- **Working branch:** `Revamp`

---

## Quick Commands

Copy-paste these prompts to Claude:

### Get Context (Start of Session)
```
Read SESSION_LOG.md and give me a summary of recent changes
```

### Code Cleanup
```
clean up the code
```
Scans for: unused imports, dead code, duplicates, undefined variables

### Run Tests
```
test
```
or
```
run tests
```
Runs: `pytest tests/ -v`

### Log Changes
```
log it
```
Appends entry to SESSION_LOG.md with date, changes, files, test outcome

### Full Check (All 3)
```
clean, test, and log
```
Runs cleanup scan, pytest, then logs results

### Retrain Base Model
```
retrain the base model
```
Retrains `models/base_model.pkl` with `training_data/base_training_data.csv`

### Add Training Data
```
add more [names/countries/phones/etc] to training data
```
Expands `training_data/base_training_data.csv` and retrains

---

## Session Log Workflow

**IMPORTANT:** This project uses `SESSION_LOG.md` to track progress.

### At the start of each session:
1. Read `SESSION_LOG.md` to understand recent changes and context

### During the session:
1. Make changes as requested
2. User tests the changes
3. When user says **"log it"** (or similar), append an entry to `SESSION_LOG.md` with:
   - Date
   - What was changed
   - Key files modified
   - Test outcome / Notes

### Entry format:
```markdown
### YYYY-MM-DD
- **Changes:** What was modified
- **Files:** Key files changed
- **Tested:** Yes/No - outcome
- **Notes:** Any additional context
```

## Key Files
- `app.py` - Main Streamlit application
- `ml/` - Core ML modules (validator.py, corrector.py, feature_extractor.py)
- `models/` - Trained ML models
- `training_data/` - Training CSV files
- `reference_lists/` - Valid values for finite columns (country, age, etc.)
- `tests/` - Pytest test suite (30 tests)
- `SESSION_LOG.md` - Progress tracking log
- `MODEL_GUIDE.md` - How to improve model accuracy

## Sub-Agent Workflows

This project uses 3 specialized sub-agents. Run them individually or in parallel.

### Available Agents:

| Command | Agent | What it does |
|---------|-------|--------------|
| "clean up the code" | Code Cleanup | Find unused imports, dead code, duplicates, complexity issues |
| "run tests" / "check tests" | Testing | Run tests or check testing infrastructure |
| "log it" | Logging | Append entry to SESSION_LOG.md |

### Parallel Execution:
Say **"do all three in parallel"** or **"run cleanup, tests, and log in parallel"** to execute multiple agents simultaneously.

### After Making Changes:
1. User tests the changes manually
2. User says "log it" to record the changes
3. Optionally run cleanup agent to check code quality

### Known Issues (updated 2026-02-05):
- ~~`self.typo_map` undefined in `ml/corrector.py:81`~~ ✅ Fixed
- ~~Unused imports~~ ✅ Fixed (moved to top-level)
- ~~No automated test infrastructure~~ ✅ Fixed (30 tests in `tests/`)
- Print statements should be replaced with logging (low priority)

## Recommended Workflow

**DO NOT auto-run all agents after every change.** Keep the user in control.

### Normal workflow:
1. User requests a change
2. Claude makes the change
3. User tests manually
4. User says "log it" → Claude logs to SESSION_LOG.md

### When to run agents:

| Situation | User says |
|-----------|-----------|
| Small change, tested, ready to move on | "log it" |
| Done for the day, want full check | "cleanup, test, and log" |
| Major feature complete | "do all 3 in parallel" |
| Just curious about code health | "clean up the code" |

### Why not auto-run everything:
- Overkill for small fixes
- User must test before logging
- Cleanup may find unrelated issues
- Wastes time on trivial changes
