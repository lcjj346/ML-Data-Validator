"""
ML Data Validator - Streamlit Web Interface

A general-purpose ML-based data validation system with plugin architecture.
Automatically detects and validates multiple data types including phone numbers,
emails, dates, and numeric ranges (blood sugar, height, weight, calories, etc.).
"""

import streamlit as st
import pandas as pd
import os
import io
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Import the new plugin system
from ml.init_validators import initialize_validators, get_validator_status
from ml.validator_registry import get_global_registry
from ml.column_type_detector import ColumnTypeDetector

st.set_page_config(page_title="ML-Data-Validator", layout="wide", page_icon="✓")

# ==================== INITIALIZATION ====================

# Initialize the validator registry (only once)
if 'registry' not in st.session_state:
    with st.spinner("Initializing ML validators..."):
        st.session_state.registry = initialize_validators()
        st.session_state.detector = ColumnTypeDetector()
        st.session_state.validator_status = get_validator_status()

registry = st.session_state.registry
detector = st.session_state.detector

# ==================== HEADER ====================

st.title("🔍 ML Data Validator")
st.markdown("**General-purpose ML-based data validation for any CSV/Excel file**")

# ==================== TABS ====================

tab_validation, tab_training, tab_status = st.tabs([
    "📊 Data Validation",
    "🎓 Model Training",
    "ℹ️ System Status"
])

# ==================== SYSTEM STATUS TAB ====================

with tab_status:
    st.header("System Status")

    status = st.session_state.validator_status

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Validators", status['total_validators'])
    with col2:
        st.metric("Total Correctors", status['total_correctors'])
    with col3:
        total_types = len(status['types'])
        st.metric("Supported Data Types", total_types)

    st.subheader("Registered Data Types")

    # Create a nice table
    data_types_list = []
    for data_type, info in sorted(status['types'].items()):
        data_types_list.append({
            'Data Type': data_type,
            'Validator': '✓' if info['has_validator'] else '✗',
            'Corrector': '✓' if info['has_corrector'] else '✗',
            'Status': 'Ready' if (info['has_validator'] or info['has_corrector']) else 'Not Available'
        })

    st.dataframe(pd.DataFrame(data_types_list), use_container_width=True, hide_index=True)

    with st.expander("📖 Supported Data Types Documentation"):
        st.markdown("""
        ### Phone Numbers
        - **Validator**: Logistic Regression ML model
        - **Corrector**: XGBoost character-level edit distance
        - **Use case**: Validate and correct phone numbers with country codes

        ### Email Addresses
        - **Validator**: RFC-compliant pattern matching
        - **Use case**: Validate email addresses and detect common typos

        ### Numeric Ranges
        - **Blood Sugar** (70-180 mg/dL typical range)
        - **Height** (140-210 cm typical range)
        - **Weight** (30-200 kg typical range)
        - **Calories** (1000-4000 kcal typical range)
        - **Heart Rate** (60-100 bpm typical range)
        - **Steps** (1000-20000 steps typical range)
        - **Temperature** (36-38°C typical range)
        - **Blood Pressure** (systolic/diastolic)
        - **Age** (0-120 years typical range)

        ### Dates
        - **Validator**: Multi-format date parser
        - **Use case**: Validate dates in various formats (YYYY-MM-DD, DD/MM/YYYY, etc.)
        """)

# ==================== VALIDATION TAB ====================

with tab_validation:
    st.header("Data Validation")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xlsx"],
        help="Upload a file with any columns - the system will automatically detect data types"
    )

    if uploaded_file:
        try:
            # Read file
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✓ File uploaded successfully! Shape: {df.shape}")

            # Auto-detect column types
            with st.spinner("Detecting column types..."):
                detected_types = detector.detect_all_columns(df)

            # Show detected types
            with st.expander("🔍 Auto-Detected Column Types", expanded=True):
                col_info = []
                for col_name, data_type in detected_types.items():
                    has_validator = registry.has_validator(data_type)
                    has_corrector = registry.has_corrector(data_type)

                    col_info.append({
                        'Column': col_name,
                        'Detected Type': data_type,
                        'Validator': '✓' if has_validator else '✗',
                        'Corrector': '✓' if has_corrector else '✗'
                    })

                st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

            # Validation function using the plugin system
            def validate_data(dataframe, column_types):
                """Validate the data using the plugin system"""
                result_df = dataframe.copy()

                for col in dataframe.columns:
                    data_type = column_types.get(col, 'text')

                    # Get validation results from registry
                    validation_results = registry.validate_batch(data_type, dataframe[col].tolist())

                    if validation_results:
                        # Extract is_valid and confidence from ValidationResult objects
                        result_df[f"{col}_Valid"] = [r.is_valid for r in validation_results]
                        result_df[f"{col}_Confidence"] = [r.confidence for r in validation_results]
                        result_df[f"{col}_DataType"] = data_type
                    else:
                        # No validator available - use generic validation
                        def is_valid_generic(value):
                            return str(value).strip() not in ['', 'nan', 'None', 'null']

                        result_df[f"{col}_Valid"] = dataframe[col].apply(is_valid_generic)
                        result_df[f"{col}_Confidence"] = [0.5] * len(dataframe)
                        result_df[f"{col}_DataType"] = 'text'

                return result_df

            # Perform validation
            with st.spinner("Validating data..."):
                validated_df = validate_data(df, detected_types)

            # Store in session state with a flag to track updates
            if 'current_df' not in st.session_state or 'file_name' not in st.session_state or st.session_state.file_name != uploaded_file.name:
                st.session_state.current_df = validated_df.copy()
                st.session_state.original_columns = list(df.columns)
                st.session_state.column_types = detected_types
                st.session_state.file_name = uploaded_file.name
                st.session_state.data_version = 0

            # Ensure data_version exists
            if 'data_version' not in st.session_state:
                st.session_state.data_version = 0

            # ==================== DATA QUALITY DASHBOARD ====================

            st.subheader("📈 Data Quality Dashboard")

            original_columns = st.session_state.original_columns
            invalid_counts = {}

            for col in original_columns:
                invalid_counts[col] = (~st.session_state.current_df[f"{col}_Valid"]).sum()

            # Metrics
            total_records = len(st.session_state.current_df)

            # Calculate columns needed (overall + individual columns)
            num_cols = min(len(original_columns) + 1, 6)  # Limit to 6 columns max
            cols = st.columns(num_cols)

            # Overall quality
            with cols[0]:
                total_invalid = sum(invalid_counts.values())
                total_cells = len(original_columns) * total_records
                overall_quality = ((total_cells - total_invalid) / total_cells) * 100 if total_cells > 0 else 0
                st.metric("Overall Quality", f"{overall_quality:.1f}%",
                         f"{total_cells - total_invalid}/{total_cells} valid")

            # Individual column metrics (show first few)
            for i, (col, invalid_count) in enumerate(list(invalid_counts.items())[:num_cols-1], 1):
                with cols[i]:
                    accuracy = ((total_records - invalid_count) / total_records) * 100
                    avg_confidence = st.session_state.current_df[f"{col}_Confidence"].mean()
                    st.metric(
                        f"{col[:15]}...",
                        f"{accuracy:.1f}%",
                        f"{invalid_count} issues"
                    )

            # ==================== INTERACTIVE DATA TABLE ====================

            st.subheader("📝 Interactive Data Editor")
            st.write("Invalid data highlighted in red. Valid data in green. Click cells to edit.")

            # Cell styling: red for invalid, green for valid
            cell_style_jscode = JsCode('''
            function(params) {
                const field = params.colDef.field;
                const validField = field + '_Valid';

                if (params.data[validField] === false) {
                    // RED for invalid
                    return {
                        'backgroundColor': '#750E21',
                        'color': '#ffffff',
                        'fontWeight': 'bold'
                    };
                } else if (params.data[validField] === true) {
                    // Darker GREEN for valid
                    return {
                        'backgroundColor': '#1E7E34',
                        'color': '#ffffff'
                    };
                }
                return {};
            }
            ''')

            # Configure grid
            gb = GridOptionsBuilder.from_dataframe(st.session_state.current_df)

            # Make original columns editable, hide validation columns
            for col in st.session_state.current_df.columns:
                if col.endswith('_Valid') or col.endswith('_Confidence') or col.endswith('_DataType'):
                    gb.configure_column(col, hide=True)
                elif col in original_columns:
                    # Check if column should show decimals
                    col_type = detected_types.get(col, 'text')

                    # Only Height and Weight show decimals
                    if col_type in ['numeric_range:height', 'numeric_range:weight']:
                        gb.configure_column(
                            col,
                            editable=True,
                            cellStyle=cell_style_jscode,
                            type=["numericColumn", "numberColumnFilter"],
                            precision=1  # 1 decimal place
                        )
                    elif col_type.startswith('numeric_range:'):
                        # Other numeric ranges: no decimals
                        gb.configure_column(
                            col,
                            editable=True,
                            cellStyle=cell_style_jscode,
                            type=["numericColumn", "numberColumnFilter"],
                            precision=0  # No decimals
                        )
                    else:
                        # Text, phone, email, date: default formatting
                        gb.configure_column(col, editable=True, cellStyle=cell_style_jscode)

            gb.configure_grid_options(enableRangeSelection=True, enableCellTextSelection=True)
            grid_options = gb.build()

            # Display grid with versioned key to force refresh
            grid_key = f"data_grid_{st.session_state.get('data_version', 0)}"
            grid_response = AgGrid(
                st.session_state.current_df,
                gridOptions=grid_options,
                update_mode='VALUE_CHANGED',
                allow_unsafe_jscode=True,
                fit_columns_on_grid_load=True,
                height=400,
                key=grid_key
            )

            # Handle manual edits
            if grid_response['data'] is not None:
                edited_df = pd.DataFrame(grid_response['data'])

                # Re-validate after edits
                validated_edited_df = validate_data(edited_df[original_columns], detected_types)

                # Update session state
                st.session_state.current_df = validated_edited_df

                # Show update message
                changes_made = not edited_df[original_columns].equals(df)
                if changes_made:
                    st.info("ℹ️ Data has been updated. Validation refreshed automatically.")

            # ==================== ML-POWERED SUGGESTIONS ====================

            st.subheader("💡 ML-Powered Correction Suggestions")

            # Show model status
            st.info("🤖 Using plugin-based ML correctors for intelligent suggestions")

            # Generate suggestions for invalid data
            suggestions_found = False
            for col in original_columns:
                try:
                    valid_col_name = f"{col}_Valid"
                    data_type = detected_types.get(col, 'text')

                    if valid_col_name in st.session_state.current_df.columns:
                        invalid_mask = st.session_state.current_df[valid_col_name] == False
                        invalid_count = int(invalid_mask.sum())

                        if invalid_count > 0 and registry.has_corrector(data_type):
                            suggestions_found = True
                            expander_title = f"{col} - {invalid_count} issues found ({data_type})"

                            with st.expander(expander_title, expanded=True):
                                invalid_data = st.session_state.current_df[invalid_mask]
                                suggestions = []

                                for idx, row in invalid_data.iterrows():
                                    try:
                                        original_value = row[col]

                                        # Get correction from registry
                                        correction_result = registry.correct_value(data_type, original_value)

                                        if correction_result and correction_result.was_corrected():
                                            row_number = int(idx) + 1
                                            suggestions.append({
                                                'Row': row_number,
                                                'Current Value': str(original_value),
                                                'Suggested Fix': str(correction_result.corrected_value),
                                                'Confidence': correction_result.confidence,
                                                'Type': correction_result.correction_type or 'correction'
                                            })
                                    except Exception:
                                        continue

                                if suggestions:
                                    # Display suggestions with apply buttons
                                    for i, suggestion in enumerate(suggestions):
                                        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 1, 1])

                                        with col1:
                                            st.write(f"**Row {suggestion['Row']}**")

                                        with col2:
                                            st.write(f"Current: `{suggestion['Current Value']}`")

                                        with col3:
                                            st.write(f"Suggested: `{suggestion['Suggested Fix']}`")

                                        with col4:
                                            st.write(f"{suggestion['Confidence']:.0%}")

                                        with col5:
                                            button_key = f"apply_{col}_{suggestion['Row']}_{i}_{st.session_state.get('data_version', 0)}"
                                            if st.button("Apply", key=button_key):
                                                try:
                                                    row_idx = suggestion['Row'] - 1
                                                    suggested_value = suggestion['Suggested Fix']

                                                    # Create a fresh copy to avoid dtype issues
                                                    temp_df = st.session_state.current_df[original_columns].copy()

                                                    # Convert column to string type to avoid dtype conflicts
                                                    temp_df[col] = temp_df[col].astype(str)
                                                    temp_df.at[row_idx, col] = str(suggested_value)

                                                    # Re-validate with the updated data
                                                    updated_data = validate_data(temp_df, detected_types)
                                                    st.session_state.current_df = updated_data

                                                    # Increment version to force grid refresh
                                                    st.session_state.data_version = st.session_state.get('data_version', 0) + 1

                                                    st.success(f"✓ Applied suggestion for Row {suggestion['Row']}")
                                                    st.rerun()
                                                except Exception as e:
                                                    st.error(f"Error applying suggestion: {str(e)}")

                                        if i < len(suggestions) - 1:
                                            st.markdown("---")

                                    # Add bulk apply button
                                    st.markdown("---")
                                    bulk_button_key = f"apply_all_{col}_{st.session_state.get('data_version', 0)}"
                                    if st.button(f"Apply All {len(suggestions)} Suggestions", key=bulk_button_key):
                                        try:
                                            # Create a fresh copy to avoid dtype issues
                                            temp_df = st.session_state.current_df[original_columns].copy()

                                            # Convert column to string type to avoid dtype conflicts
                                            temp_df[col] = temp_df[col].astype(str)

                                            applied_count = 0
                                            for suggestion in suggestions:
                                                row_idx = suggestion['Row'] - 1
                                                suggested_value = suggestion['Suggested Fix']
                                                temp_df.at[row_idx, col] = str(suggested_value)
                                                applied_count += 1

                                            # Re-validate all with the updated data
                                            updated_data = validate_data(temp_df, detected_types)
                                            st.session_state.current_df = updated_data

                                            # Increment version to force grid refresh
                                            st.session_state.data_version = st.session_state.get('data_version', 0) + 1

                                            st.success(f"✓ Applied {applied_count} suggestions to {col}")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error applying bulk suggestions: {str(e)}")
                                else:
                                    st.info("No automatic suggestions available for these values")

                except Exception:
                    continue

            if not suggestions_found:
                st.success("✅ No invalid data found - all data looks good!")

            # ==================== EXPORT SECTION ====================

            st.subheader("📥 Export Clean Data")

            # Prepare export data (remove validation columns)
            export_df = st.session_state.current_df[original_columns].copy()

            # Export buttons
            col1, col2 = st.columns(2)

            with col1:
                # CSV export
                csv_data = export_df.to_csv(index=False)
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    "📄 Download Clean CSV",
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
                    "📊 Download Excel",
                    output.getvalue(),
                    file_name=f"validated_data_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ==================== TRAINING TAB ====================

with tab_training:
    from ml.model_trainer import PhoneModelTrainer
    import xgboost as xgb

    st.header("🎓 Model Training Center")
    st.markdown("""
    Train or retrain ML models with your own data. Upload custom training CSV files to improve model accuracy.
    """)

    # Create sub-tabs for different models
    train_tab1, train_tab2 = st.tabs([
        "Phone Validator (Logistic Regression)",
        "Phone Corrector (XGBoost)"
    ])

    # ==================== LOGISTIC REGRESSION TRAINING ====================
    with train_tab1:
        st.subheader("Phone Number Validator (Logistic Regression)")

        st.info("""
        **Training Data Format:**
        - Column 1: `phone` - Phone number string
        - Column 2: `is_valid` - 1 for valid, 0 for invalid

        **Example:**
        ```
        phone,is_valid
        +1234567890,1
        +123,0
        abc123,0
        ```
        """)

        # File upload for training
        lr_file = st.file_uploader(
            "Upload Training Data (CSV)",
            type=['csv'],
            key='lr_upload',
            help="CSV file with 'phone' and 'is_valid' columns"
        )

        training_df_lr = None

        if lr_file:
            try:
                training_df_lr = pd.read_csv(lr_file)

                if 'phone' not in training_df_lr.columns or 'is_valid' not in training_df_lr.columns:
                    st.error("CSV must have 'phone' and 'is_valid' columns")
                    training_df_lr = None
                else:
                    st.success(f"Loaded {len(training_df_lr)} training samples")

                    with st.expander("Preview Data"):
                        st.dataframe(training_df_lr.head(20))

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", len(training_df_lr))
                        with col2:
                            valid_count = training_df_lr['is_valid'].sum()
                            st.metric("Valid Samples", valid_count)
                        with col3:
                            invalid_count = len(training_df_lr) - valid_count
                            st.metric("Invalid Samples", invalid_count)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

        # Train button
        if training_df_lr is not None:
            if st.button("Train Logistic Regression Model", type="primary", key='train_lr'):
                with st.spinner("Training in progress..."):
                    try:
                        trainer = PhoneModelTrainer()

                        # Prepare data
                        training_data = pd.DataFrame({
                            'phone': training_df_lr['phone'],
                            'is_valid': training_df_lr['is_valid']
                        })

                        st.info(f"Training on {len(training_data)} samples...")

                        # Train
                        results = trainer.train_from_data(training_data)

                        st.info("Saving model...")

                        # Save model
                        save_path = 'saved_models/phone_validator_model.pkl'
                        trainer.save_model(save_path)

                        st.success("✅ Training completed successfully!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{results['accuracy']:.1%}")
                        with col2:
                            st.metric("Model Saved", "✓")

                        st.info(f"Model saved to: `{save_path}`")
                        st.info("Refresh the page to load the new model")

                        with st.expander("Detailed Classification Report"):
                            st.json(results['classification_report'])

                        st.balloons()

                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

    # ==================== XGBOOST TRAINING ====================
    with train_tab2:
        st.subheader("Phone Number Corrector (XGBoost)")

        st.info("""
        **Training Data Format:**
        - Column 1: `invalid` - Invalid phone number
        - Column 2: `valid` - Corrected phone number
        - Column 3: `operations` - Edit operations (pipe-separated)

        **Example:**
        ```
        invalid,valid,operations
        1234567890,+11234567890,insert_country|keep|keep...
        ```
        """)

        st.warning("XGBoost training requires specifically formatted data with edit operations.")

        # File upload
        xgb_file = st.file_uploader(
            "Upload Training Data (CSV)",
            type=['csv'],
            key='xgb_upload',
            help="CSV with 'invalid', 'valid', and 'operations' columns"
        )

        training_df_xgb = None

        if xgb_file:
            try:
                training_df_xgb = pd.read_csv(xgb_file)

                required_cols = ['invalid', 'valid', 'operations']
                if not all(col in training_df_xgb.columns for col in required_cols):
                    st.error(f"CSV must have columns: {', '.join(required_cols)}")
                    training_df_xgb = None
                else:
                    st.success(f"Loaded {len(training_df_xgb)} training samples")
                    with st.expander("Preview Data"):
                        st.dataframe(training_df_xgb.head(20))
                        st.metric("Total Samples", len(training_df_xgb))
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

        # Train button
        if training_df_xgb is not None:
            if st.button("Train XGBoost Model", type="primary", key='train_xgb'):
                with st.spinner("Training in progress..."):
                    try:
                        from ml.edit_distance_corrector import EditDistanceCorrector

                        corrector = EditDistanceCorrector()

                        # Convert DataFrame to training_data format
                        training_data = []
                        for _, row in training_df_xgb.iterrows():
                            training_data.append({
                                'invalid': row['invalid'],
                                'valid': row['valid'],
                                'operations': row['operations'].split('|')
                            })

                        st.info(f"Preparing features from {len(training_data)} samples...")

                        # Prepare features
                        X, y = corrector.prepare_training_data(training_data)

                        st.info("Training XGBoost model...")

                        # Train
                        unique_labels = pd.unique(y)
                        label_map = {old: new for new, old in enumerate(unique_labels)}
                        y_remapped = pd.array([label_map[label] for label in y])

                        corrector.inference_label_map = {new: old for old, new in label_map.items()}

                        corrector.model = xgb.XGBClassifier(
                            objective='multi:softmax',
                            num_class=len(unique_labels),
                            random_state=42
                        )

                        corrector.model.fit(X, y_remapped)

                        # Evaluate
                        accuracy = corrector.model.score(X, y_remapped)

                        st.info("Saving model...")

                        # Save
                        save_path = 'saved_models/edit_distance_corrector.pkl'
                        corrector.save(save_path)

                        st.success("✅ Training completed successfully!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Training Accuracy", f"{accuracy:.1%}")
                        with col2:
                            st.metric("Model Saved", "✓")

                        st.info(f"Model saved to: `{save_path}`")
                        st.info("Refresh the page to load the new model")

                        st.balloons()

                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
