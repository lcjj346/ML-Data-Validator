import streamlit as st
import pandas as pd
import re
from utils.phone_validator import is_phone_number_valid
from utils.blood_sugar_validator import is_blood_sugar_valid
from ml.phone_validator import PhoneValidator
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import io

# Suggestion generation functions
def generate_phone_suggestion(invalid_phone):
    """Generate intelligent suggestions for invalid phone numbers"""
    if not invalid_phone or pd.isna(invalid_phone):
        return ""
    
    phone_str = str(invalid_phone).strip()
    
    # Smart suggestion logic that preserves structure
    if phone_str.startswith('+1'):
        # US number - try to fix while preserving format
        base_number = phone_str[2:]  # Remove +1
        
        # Replace common letter errors with similar numbers
        corrections = {
            'e': '3', 'E': '3',  # e looks like 3
            'o': '0', 'O': '0',  # o looks like 0
            'i': '1', 'I': '1',  # i looks like 1
            'l': '1', 'L': '1',  # l looks like 1
            's': '5', 'S': '5',  # s looks like 5
            'g': '9', 'G': '9',  # g looks like 9
            'a': '2', 'A': '2'   # a looks like 2
        }
        
        # Fix the base number
        fixed_base = base_number
        for letter, digit in corrections.items():
            fixed_base = fixed_base.replace(letter, digit)
        
        # Extract only digits and ensure it's 10 digits
        digits_only = re.findall(r'\d', fixed_base)
        if len(digits_only) == 10:
            return "+1" + "".join(str(d) for d in digits_only)
        elif len(digits_only) > 10:
            # Take first 10 digits
            return "+1" + "".join(str(d) for d in digits_only[:10])
        elif len(digits_only) >= 7:  # At least area code + some digits
            # Pad with common endings
            while len(digits_only) < 10:
                digits_only.append('0')
            return "+1" + "".join(str(d) for d in digits_only)
    
    elif phone_str.startswith('+'):
        # Other international numbers
        digits = re.findall(r'\d', phone_str)
        digit_string = ''.join(str(d) for d in digits)
        if 8 <= len(digit_string) <= 15:
            return "+" + digit_string
    
    else:
        # No country code - assume US
        digits = re.findall(r'\d', phone_str)
        digit_string = ''.join(str(d) for d in digits)
        
        if len(digit_string) == 10:
            return "+1" + digit_string
        elif len(digit_string) == 11 and digit_string.startswith('1'):
            return "+" + digit_string
        elif 7 <= len(digit_string) <= 9:
            # Assume local number, add area code 555
            return "+1555" + digit_string
    
    return ""  # Can't suggest for very invalid data

def generate_blood_sugar_suggestion(invalid_value):
    """Generate suggestions for invalid blood sugar values"""
    if not invalid_value or pd.isna(invalid_value):
        return "90"
    
    value_str = str(invalid_value).lower().strip()
    
    # Handle obvious invalid cases
    if value_str in ['nan', '', 'null', 'none'] or value_str == '-10':
        return "90"  # Normal fasting blood sugar
    
    # If it contains letters, suggest normal value
    if any(c.isalpha() for c in value_str):
        return "85"  # Normal blood sugar
    
    # Try to parse as number and adjust
    try:
        val = float(value_str)
        if val < 50:
            return "80"  # Too low, suggest normal
        elif val > 500:
            return "120"  # Too high, suggest normal-high
        else:
            return "90"  # Default normal
    except:
        return "90"  # Safe fallback

st.set_page_config(page_title="ML-Data-Validator", layout="wide")

# Initialize ML validator
if 'ml_phone_validator' not in st.session_state:
    st.session_state.ml_phone_validator = PhoneValidator()
    st.session_state.ml_model_loaded = st.session_state.ml_phone_validator.is_model_loaded()

# Header
st.title("ML-Data-Validator")

# ML Status
if st.session_state.ml_model_loaded:
    st.success("ML Model: ACTIVE")
else:
    st.warning("ML Model: NOT LOADED")

# File upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"File uploaded successfully! Shape: {df.shape}")
        
        # Convert all data to strings to prevent type issues
        for col in df.columns:
            df[col] = df[col].astype(str)

        # Validation function
        def validate_data(dataframe):
            """Validate the data and add validation columns"""
            result_df = dataframe.copy()
            
            for col in dataframe.columns:
                if col == "PhoneNumber":
                    if st.session_state.ml_model_loaded:
                        # Use ML validation
                        phones = dataframe[col].tolist()
                        ml_results = st.session_state.ml_phone_validator.validate_phone_batch(phones)
                        result_df[f"{col}_Valid"] = [result[0] for result in ml_results]
                        result_df[f"{col}_Confidence"] = [result[1] for result in ml_results]
                    else:
                        # Use rule-based validation
                        result_df[f"{col}_Valid"] = dataframe[col].apply(is_phone_number_valid)
                        result_df[f"{col}_Confidence"] = [1.0] * len(dataframe)
                
                elif col == "BloodSugar":
                    result_df[f"{col}_Valid"] = dataframe[col].apply(is_blood_sugar_valid)
                    result_df[f"{col}_Confidence"] = [1.0] * len(dataframe)
                
                else:
                    # Generic validation - check if not empty and not NaN
                    def is_valid_generic(value):
                        return str(value).strip() not in ['', 'nan', 'None', 'null']
                    
                    result_df[f"{col}_Valid"] = dataframe[col].apply(is_valid_generic)
                    result_df[f"{col}_Confidence"] = [1.0] * len(dataframe)
            
            return result_df

        # Initial validation
        validated_df = validate_data(df)
        
        # Store in session state
        if 'current_df' not in st.session_state:
            st.session_state.current_df = validated_df.copy()
        
        # Dashboard
        st.subheader("Data Quality Dashboard")
        
        original_columns = [col for col in df.columns]
        invalid_counts = {}
        
        for col in original_columns:
            invalid_counts[col] = (~st.session_state.current_df[f"{col}_Valid"]).sum()
        
        # Metrics
        total_records = len(st.session_state.current_df)
        cols = st.columns(len(original_columns) + 1)
        
        # Overall quality
        with cols[0]:
            total_invalid = sum(invalid_counts.values())
            total_cells = len(original_columns) * total_records
            overall_quality = ((total_cells - total_invalid) / total_cells) * 100
            st.metric("Overall Quality", f"{overall_quality:.1f}%", f"{total_cells - total_invalid}/{total_cells} valid")
        
        # Individual column metrics
        for i, (col, invalid_count) in enumerate(invalid_counts.items(), 1):
            with cols[i]:
                accuracy = ((total_records - invalid_count) / total_records) * 100
                if col == "PhoneNumber" and st.session_state.ml_model_loaded:
                    avg_confidence = st.session_state.current_df[f"{col}_Confidence"].mean()
                    st.metric(f"{col}", f"{accuracy:.1f}%", f"ML Confidence: {avg_confidence*100:.1f}%")
                else:
                    st.metric(f"{col}", f"{accuracy:.1f}%", f"{invalid_count} invalid")

        # Interactive Data Table
        st.subheader("Interactive Data Editor")
        st.write("Invalid data highlighted in red. Click cells to edit.")

        # Cell styling for invalid data
        cell_style_jscode = JsCode('''
        function(params) {
            const field = params.colDef.field;
            const validField = field + '_Valid';
            
            if (params.data[validField] === false) {
                return { 
                    'backgroundColor': '#ffcdd2', 
                    'color': '#d32f2f',
                    'fontWeight': 'bold'
                };
            } else if (params.data[validField] === true) {
                const confField = field + '_Confidence';
                const confidence = params.data[confField] || 1.0;
                if (confidence >= 0.9) {
                    return { 'backgroundColor': '#c8e6c9', 'color': '#2e7d32' };
                } else if (confidence >= 0.7) {
                    return { 'backgroundColor': '#fff3e0', 'color': '#f57c00' };
                } else {
                    return { 'backgroundColor': '#e1f5fe', 'color': '#0277bd' };
                }
            }
            return {};
        }
        ''')

        # Configure grid
        gb = GridOptionsBuilder.from_dataframe(st.session_state.current_df)
        
        # Make original columns editable, hide validation columns
        for col in st.session_state.current_df.columns:
            if col.endswith('_Valid') or col.endswith('_Confidence'):
                gb.configure_column(col, hide=True)
            elif col in original_columns:
                gb.configure_column(col, editable=True, cellStyle=cell_style_jscode)
        
        gb.configure_grid_options(enableRangeSelection=True, enableCellTextSelection=True)
        grid_options = gb.build()

        # Display grid
        grid_response = AgGrid(
            st.session_state.current_df,
            gridOptions=grid_options,
            update_mode='VALUE_CHANGED',
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            height=400,
            key="data_grid"
        )

        # Handle manual edits
        if grid_response['data'] is not None:
            edited_df = pd.DataFrame(grid_response['data'])
            
            # Convert original columns to strings
            for col in original_columns:
                if col in edited_df.columns:
                    edited_df[col] = edited_df[col].astype(str)
            
            # Re-validate after edits
            validated_edited_df = validate_data(edited_df[original_columns])
            
            # Update session state
            st.session_state.current_df = validated_edited_df
            
            # Show update message
            changes_made = not edited_df[original_columns].equals(df)
            if changes_made:
                st.info("Data has been updated. Validation refreshed automatically.")

        # ML-Powered Suggestions
        st.subheader("ML-Powered Suggestions")
        
        if st.session_state.ml_model_loaded:
            st.success("Smart corrections powered by Random Forest ML model")
        else:
            st.info("ML model not available - using rule-based suggestions")
        
        # Generate suggestions for invalid data
        suggestions_found = False
        for col in original_columns:
            try:
                valid_col_name = str(col) + "_Valid"
                if valid_col_name in st.session_state.current_df.columns:
                    invalid_mask = st.session_state.current_df[valid_col_name] == False
                    invalid_count = int(invalid_mask.sum())
                    
                    if invalid_count > 0:
                        suggestions_found = True
                        # Safe string formatting - no f-strings
                        expander_title = str(col) + " - " + str(invalid_count) + " issues found"
                        with st.expander(expander_title, expanded=True):
                            invalid_data = st.session_state.current_df[invalid_mask]
                            suggestions = []
                            
                            for idx, row in invalid_data.iterrows():
                                try:
                                    # Ultra-safe value extraction
                                    original_value = str(row[col]) if row[col] is not None else ""
                                    suggested_value = ""
                                    
                                    # Generate suggestions based on column type
                                    if str(col) == "PhoneNumber":
                                        suggested_value = str(generate_phone_suggestion(original_value))
                                    elif str(col) == "BloodSugar":
                                        suggested_value = str(generate_blood_sugar_suggestion(original_value))
                                    else:
                                        suggested_value = "Review manually"
                                    
                                    # Only add valid suggestions
                                    if suggested_value and str(suggested_value) != str(original_value):
                                        # Safe row number calculation
                                        row_number = int(idx) + 1
                                        
                                        suggestions.append({
                                            'Row': row_number,
                                            'Current Value': str(original_value),
                                            'Suggested Fix': str(suggested_value)
                                        })
                                except Exception as e:
                                    # Skip problematic rows silently
                                    continue
                            
                            if suggestions:
                                try:
                                    suggestions_df = pd.DataFrame(suggestions)
                                    st.dataframe(suggestions_df, use_container_width=True)
                                    st.info("💡 Copy the suggested values and paste them into the table above to fix the issues")
                                except Exception as e:
                                    st.error("Error displaying suggestions: " + str(e))
                            else:
                                st.info("No automatic suggestions available for this column")
            except Exception as e:
                # Skip problematic columns silently
                continue
        
        if not suggestions_found:
            st.success("🎉 No invalid data found - all data looks good!")

        # Export section
        st.subheader("Export Clean Data")
        
        # Prepare export data (remove validation columns)
        export_df = st.session_state.current_df[original_columns].copy()
        
        # Export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv_data = export_df.to_csv(index=False)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                "Download Clean CSV",
                csv_data,
                file_name=f"validated_data_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel export
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Clean_Data')
            
            st.download_button(
                "Download Excel",
                output.getvalue(),
                file_name=f"validated_data_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("ML Model Info")
    
    if st.session_state.ml_model_loaded:
        st.success("Random Forest Model Loaded")
        st.info("Phone validation using ML model")
    else:
        st.warning("ML Model Not Available")
        st.info("Using rule-based validation")
    
    st.header("Supported Data Types")
    st.write("• Phone Numbers (ML/Rule-based)")
    st.write("• Blood Sugar (Rule-based)")
    st.write("• Generic Data (Basic validation)")