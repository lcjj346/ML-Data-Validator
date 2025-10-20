# Troubleshooting Guide

## Issue: Apply Button Doesn't Update Data Grid

### Problem
When clicking "Apply" on correction suggestions, the data in the interactive table doesn't change visually, even though the correction was applied to the underlying data.

### Root Cause
Streamlit's AgGrid component doesn't automatically refresh when session state changes. The grid needs a new unique key to force a re-render.

### Solution ✅ (Fixed in latest version)

**What was changed:**

1. **Added Version Tracking**
```python
# Initialize version counter
st.session_state.data_version = 0

# Increment version after each update
st.session_state.data_version = st.session_state.get('data_version', 0) + 1
```

2. **Versioned Grid Key**
```python
# Old (doesn't refresh):
grid_response = AgGrid(..., key="data_grid")

# New (forces refresh):
grid_key = f"data_grid_{st.session_state.get('data_version', 0)}"
grid_response = AgGrid(..., key=grid_key)
```

3. **Versioned Button Keys**
```python
# Old (can cause stale state):
button_key = f"apply_{col}_{row}"

# New (unique per version):
button_key = f"apply_{col}_{row}_{st.session_state.get('data_version', 0)}"
```

4. **Better Cell Update Method**
```python
# Old:
st.session_state.current_df.loc[row_idx, col] = value

# New (more reliable):
st.session_state.current_df.at[row_idx, col] = value
```

### How to Verify It's Fixed

1. Upload `test_data_multitype.csv`
2. Look for row with `50000` in Calories column (should be red/invalid)
3. Click "Apply" on the suggestion that says `50000 → 5000`
4. The grid should refresh and show:
   - ✅ Cell now shows `5000`
   - ✅ Cell turns green (valid)
   - ✅ Suggestion disappears from the list

### If Still Not Working

**Check these:**

1. **Clear browser cache**
```bash
# In browser: Ctrl+Shift+R (hard refresh)
# Or: Clear cookies and cache
```

2. **Restart Streamlit**
```bash
# Stop the app (Ctrl+C)
# Start again
streamlit run app.py
```

3. **Check session state**
```python
# Add debug output (temporary):
st.write("Data version:", st.session_state.get('data_version', 0))
st.write("Grid key:", grid_key)
```

4. **Verify data is actually updating**
```python
# After clicking Apply, check:
st.write(st.session_state.current_df)
```

---

## Issue: Date Format Warning

### Problem
```
UserWarning: Could not infer format, so each element will be parsed individually,
falling back to `dateutil`. To ensure parsing is consistent and as-expected,
please specify a format.
```

### Root Cause
Pandas can't automatically determine the date format and falls back to slow parsing.

### Solution ✅ (Fixed in latest version)

The column type detector now:
1. Tries common formats first (`YYYY-MM-DD`, `DD/MM/YYYY`, etc.)
2. Suppresses warnings for flexible parsing fallback

```python
# Try specific formats first
for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']:
    date_values = pd.to_datetime(sample, format=fmt, errors='coerce')
    if date_values.notna().sum() / len(sample) > 0.7:
        return 'date'

# Fallback with suppressed warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    date_values = pd.to_datetime(sample, errors='coerce')
```

### Impact
- ⚠️ Warning is cosmetic only - doesn't affect functionality
- ✅ Now suppressed in latest version
- ✅ Date validation still works correctly

---

## Issue: Corrections Not Showing Up

### Problem
Invalid data detected but no correction suggestions appear.

### Possible Causes

1. **No corrector available for that data type**

Check System Status tab:
```
Data Type              | Validator | Corrector
-------------------------------------------------
text                   | ✗         | ✗         ← No corrector
email                  | ✓         | ✗         ← No corrector
phone                  | ✓         | ✓         ← Has corrector
numeric_range:calories | ✓         | ✓         ← Has corrector
```

**Solution:** Only phone numbers and numeric ranges have correctors currently.

2. **Value is too corrupted to correct**

Example:
```python
# Can correct:
"50000" (calories) → "5000" (decimal shift)
"1234567890" (phone) → "+11234567890" (add country code)

# Cannot correct:
"abc123xyz" (phone) → Too corrupted
"-999999" (calories) → Outside any reasonable range
```

3. **Model not loaded**

Check initialization output:
```
[FAIL] Phone Corrector: Not loaded (train model first)
```

**Solution:** Train the corrector model (see Model Training tab).

---

## Issue: Wrong Column Type Detected

### Problem
System detects wrong data type for a column.

### Examples

**Case 1: Column name is ambiguous**
```csv
Data,Value,Amount
100,200,300
```
All detected as `numeric_range:generic` (correct but not specific).

**Solution:** Rename columns to be more descriptive:
```csv
BloodSugar,Height,Calories
100,180,2000
```

**Case 2: Mixed data in column**
```csv
PhoneNumber
+1234567890
text here
+6591234567
```
May not detect as phone if too many non-phone values.

**Solution:** Clean data first or use 70%+ valid examples.

**Case 3: Similar value ranges**
```csv
Height,Age
170,70
165,65
180,80
```
Both numeric ranges (140-210 for height, 0-120 for age).

System uses column names to distinguish:
- "Height" → `numeric_range:height`
- "Age" → `numeric_range:age`

---

## Issue: Grid Won't Update After Manual Edits

### Problem
After editing a cell directly in the grid, changes don't validate.

### Root Cause
Grid's `update_mode='VALUE_CHANGED'` should trigger validation, but may have timing issues.

### Solution
The system re-validates on every grid change:
```python
if grid_response['data'] is not None:
    edited_df = pd.DataFrame(grid_response['data'])
    validated_edited_df = validate_data(edited_df[original_columns], detected_types)
    st.session_state.current_df = validated_edited_df
```

**If still not working:**
1. Make the edit
2. Click outside the cell
3. Click "Apply" on any suggestion to force refresh
4. Or upload the file again

---

## Issue: Model Not Found Errors

### Problem
```
[FAIL] Phone Validator: Not loaded (train model first)
```

### Solution

**For Phone Validator:**
1. Go to "Model Training" tab
2. Select "Phone Validator (Logistic Regression)"
3. Use existing data or upload training CSV
4. Click "Train Logistic Regression Model"
5. Wait for training to complete
6. Model automatically reloads

**For Phone Corrector:**
1. Go to "Model Training" tab
2. Select "Phone Corrector (XGBoost)"
3. Use existing data (requires special format)
4. Click "Train XGBoost Model"
5. Model automatically reloads

**For Numeric Validators:**
These work without training (use rule-based + Isolation Forest).
Can be trained for better accuracy.

---

## Issue: Large Files Slow to Process

### Problem
Files with 10,000+ rows take long to validate.

### Current Performance
- Small (< 1,000 rows): < 1 second
- Medium (1,000-10,000 rows): 1-5 seconds
- Large (10,000-100,000 rows): 5-30 seconds
- Very large (> 100,000 rows): 30+ seconds

### Optimization Tips

1. **Process in chunks** (for very large files)
```python
# Sample first 10,000 rows for validation
df_sample = df.head(10000)
```

2. **Disable unnecessary validators**
Only register validators you need.

3. **Use batch processing**
Already implemented - validators process entire columns at once.

---

## Getting Help

If you encounter an issue not covered here:

1. **Check the console/terminal** for error messages
2. **Look at the System Status tab** to verify models are loaded
3. **Try with test_data_multitype.csv** to isolate the issue
4. **Check file format** (CSV should be UTF-8 encoded)

## Debug Mode

To enable debug output:
```python
# Add to app.py temporarily
st.write("Debug Info:")
st.write("Session state keys:", list(st.session_state.keys()))
st.write("Data version:", st.session_state.get('data_version', 0))
st.write("Current DF shape:", st.session_state.current_df.shape if 'current_df' in st.session_state else "None")
```
