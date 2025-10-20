import streamlit as st
import pandas as pd
import os
import io
from ml.validator import PhoneValidator
from utils.phone_corrector import generate_phone_suggestion
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

st.set_page_config(page_title="ML-Data-Validator", layout="wide")

# Initialize ML validator
if 'ml_phone_validator' not in st.session_state:
    try:
        # Use absolute path for Streamlit app
        st.session_state.ml_phone_validator = PhoneValidator('saved_models/phone_validator_model.pkl')
        st.session_state.ml_model_loaded = st.session_state.ml_phone_validator.is_model_loaded()
        if st.session_state.ml_model_loaded:
            st.session_state.model_load_error = None
        else:
            st.session_state.model_load_error = "Model file exists but failed to load"
    except Exception as e:
        st.session_state.ml_phone_validator = None
        st.session_state.ml_model_loaded = False
        st.session_state.model_load_error = f"Failed to initialize validator: {str(e)}"

# Initialize XGBoost Edit Distance Corrector
if 'edit_distance_corrector' not in st.session_state:
    try:
        from ml.edit_distance_corrector import EditDistanceCorrector
        st.session_state.edit_distance_corrector = EditDistanceCorrector()
        corrector_loaded = st.session_state.edit_distance_corrector.load('saved_models/edit_distance_corrector.pkl')
        st.session_state.edit_distance_loaded = corrector_loaded
        if corrector_loaded:
            st.session_state.corrector_load_error = None
        else:
            st.session_state.corrector_load_error = "XGBoost corrector model not found"
    except Exception as e:
        st.session_state.edit_distance_corrector = None
        st.session_state.edit_distance_loaded = False
        st.session_state.corrector_load_error = f"Failed to load XGBoost corrector: {str(e)}"

# Header
st.title("ML-Data-Validator")

# Create tabs for main interface
tab_validation, tab_training = st.tabs(["Data Validation", "Model Training"])

# ==================== VALIDATION TAB ====================
with tab_validation:
    # ML Status
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        if st.session_state.ml_model_loaded:
            st.success("Logistic Regression (Validator): ACTIVE")
        else:
            st.warning("Logistic Regression (Validator): NOT LOADED")

    with col_status2:
        if st.session_state.get('edit_distance_loaded', False):
            st.success("XGBoost (Corrector): ACTIVE - Used for suggestions")
        else:
            st.info("XGBoost (Corrector): NOT LOADED - Using rule-based fallback")

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
                            # Use ML validation (logistic regression)
                            phones = dataframe[col].tolist()
                            ml_results = st.session_state.ml_phone_validator.validate_phone_batch(phones)
                            result_df[f"{col}_Valid"] = [result[0] for result in ml_results]
                            result_df[f"{col}_Confidence"] = [result[1] for result in ml_results]
                        else:
                            # No model available - mark all as invalid
                            st.error("Logistic regression model not loaded! Please train the model first.")
                            result_df[f"{col}_Valid"] = [False] * len(dataframe)
                            result_df[f"{col}_Confidence"] = [0.0] * len(dataframe)
                    
    # Removed BloodSugar validation - focusing on phone validation only
                    
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
                        'backgroundColor': '#750E21', 
                        'color': '#ffffff',
                        'fontWeight': 'bold'
                    };
                } else if (params.data[validField] === true) {
                    const confField = field + '_Confidence';
                    const confidence = params.data[confField] || 1.0;
                    if (confidence >= 0.9) {
                        return { 'backgroundColor': '#1F7D53', 'color': '#ffffff' };
                    } else if (confidence >= 0.7) {
                        return { 'backgroundColor': '#DCA06D', 'color': '#ffffff' };
                    } else {
                        return { 'backgroundColor': '#DCA06D', 'color': '#ffffff' };
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

            # Show which model is being used for suggestions
            if st.session_state.get('edit_distance_loaded', False):
                st.success("Smart corrections powered by XGBoost ML model (character-level editing)")
            elif st.session_state.ml_model_loaded:
                st.info("Using rule-based corrections (XGBoost model not loaded)")
            else:
                st.warning("ML models not loaded - train the models first")
            
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
                                            corrector = st.session_state.get('edit_distance_corrector')
                                            suggested_value = str(generate_phone_suggestion(original_value, corrector))
                                        else:
                                            suggested_value = "Review manually"
                                        
                                        # Only add valid suggestions
                                        # Check if suggestion is valid and different from original
                                        if suggested_value and suggested_value != "None" and str(suggested_value) != str(original_value):
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
                                        # Display suggestions with apply buttons
                                        for i, suggestion in enumerate(suggestions):
                                            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                                            
                                            with col1:
                                                st.write(f"**Row {suggestion['Row']}**")
                                            
                                            with col2:
                                                st.write(f"Current: `{suggestion['Current Value']}`")
                                            
                                            with col3:
                                                st.write(f"Suggested: `{suggestion['Suggested Fix']}`")
                                            
                                            with col4:
                                                # Create unique button key
                                                button_key = f"apply_{col}_{suggestion['Row']}_{i}"
                                                if st.button("Apply", key=button_key):
                                                    try:
                                                        # Find the actual row index in the dataframe
                                                        row_idx = suggestion['Row'] - 1  # Convert to 0-based index
                                                        suggested_value = suggestion['Suggested Fix']
                                                        
                                                        # Apply the suggestion to session state dataframe
                                                        st.session_state.current_df.loc[row_idx, col] = str(suggested_value)
                                                        
                                                        # Re-validate the updated data
                                                        updated_data = validate_data(st.session_state.current_df[original_columns])
                                                        st.session_state.current_df = updated_data
                                                        
                                                        st.success(f"Applied suggestion for Row {suggestion['Row']}")
                                                        st.rerun()
                                                        
                                                    except Exception as e:
                                                        st.error(f"Error applying suggestion: {str(e)}")
                                            
                                            # Add separator line
                                            if i < len(suggestions) - 1:
                                                st.markdown("---")
                                        
                                        # Add bulk apply button
                                        st.markdown("---")
                                        bulk_button_key = f"apply_all_{col}"
                                        if st.button(f"Apply All {col} Suggestions", key=bulk_button_key):
                                            try:
                                                applied_count = 0
                                                for suggestion in suggestions:
                                                    row_idx = suggestion['Row'] - 1
                                                    suggested_value = suggestion['Suggested Fix']
                                                    
                                                    # Apply each suggestion
                                                    st.session_state.current_df.loc[row_idx, col] = str(suggested_value)
                                                    applied_count += 1
                                                
                                                # Re-validate all updated data
                                                updated_data = validate_data(st.session_state.current_df[original_columns])
                                                st.session_state.current_df = updated_data
                                                
                                                st.success(f"Applied {applied_count} suggestions to {col}")
                                                st.rerun()
                                                
                                            except Exception as e:
                                                st.error(f"Error applying bulk suggestions: {str(e)}")
                                                
                                    except Exception as e:
                                        st.error("Error displaying suggestions: " + str(e))
                                else:
                                    st.info("No automatic suggestions available for this column")
                except Exception as e:
                    # Skip problematic columns silently
                    continue
            
            if not suggestions_found:
                st.success("No invalid data found - all data looks good!")
    
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

# ==================== TRAINING TAB ====================
with tab_training:
    from ml.model_trainer import PhoneModelTrainer
    import xgboost as xgb

    st.header("Model Training Center")
    st.markdown("""
    Train or retrain ML models with your own data. By default, models will train on existing data in the `data/` folder.
    You can also upload your own custom training data.
    """)

    # Create sub-tabs for different models
    train_tab1, train_tab2 = st.tabs([
        "Logistic Regression (Validator)",
        "XGBoost (Corrector)"
    ])

    # ====================  LOGISTIC REGRESSION TRAINING ====================
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
        invalid_phone,0
        ```
        """)

        # Data source selection
        data_source = st.radio(
            "Choose training data source:",
            ["Use existing data (Recommended)", "Upload custom data"],
            key="lr_data_source"
        )

        training_df_lr = None

        if data_source == "Use existing data (Recommended)":
            # Check for existing training data
            default_path = 'data/logistic_regression_training.csv'

            found_data = None
            if os.path.exists(default_path):
                found_data = default_path

            if found_data:
                st.success(f"Found training data: `{found_data}`")
                try:
                    training_df_lr = pd.read_csv(found_data)

                    # Validate columns
                    if 'phone' in training_df_lr.columns and 'is_valid' in training_df_lr.columns:
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
                    elif 'PhoneNumber' in training_df_lr.columns and 'Valid' in training_df_lr.columns:
                        # Convert old format to new format
                        training_df_lr = pd.DataFrame({
                            'phone': training_df_lr['PhoneNumber'],
                            'is_valid': training_df_lr['Valid']
                        })
                        st.info("Converted training data format")
                        with st.expander("Preview Data"):
                            st.dataframe(training_df_lr.head(20))
                    else:
                        st.error("CSV must have 'phone' and 'is_valid' columns (or 'PhoneNumber' and 'Valid')")
                        training_df_lr = None
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            else:
                st.warning("No training data found. Please run `python ml/generate_aligned_training_data.py` first, or upload custom data.")

        else:  # Upload custom data
            lr_file = st.file_uploader(
                "Upload Training Data (CSV)",
                type=['csv'],
                key='lr_upload',
                help="CSV file with 'phone' and 'is_valid' columns"
            )

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
                progress_placeholder = st.empty()

                with st.spinner("Training in progress..."):
                    try:
                        trainer = PhoneModelTrainer()

                        # Prepare data
                        training_data = pd.DataFrame({
                            'phone': training_df_lr['phone'],
                            'is_valid': training_df_lr['is_valid']
                        })

                        progress_placeholder.info(f"Training on {len(training_data)} samples...")

                        # Train
                        results = trainer.train_from_data(training_data)

                        progress_placeholder.info("Saving model...")

                        # Save model
                        save_path = 'saved_models/phone_validator_model.pkl'
                        trainer.save_model(save_path)

                        progress_placeholder.empty()

                        st.success("Training completed successfully!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Accuracy", f"{results['accuracy']:.1%}")
                        with col2:
                            st.metric("Model Saved", "Yes")

                        st.info(f"Model saved to: `{save_path}`")
                        st.info("Please refresh the page to load the new model")

                        with st.expander("Detailed Classification Report"):
                            st.json(results['classification_report'])

                        st.balloons()

                        # Reload validator
                        st.session_state.ml_phone_validator = PhoneValidator(save_path)
                        st.session_state.ml_model_loaded = st.session_state.ml_phone_validator.is_model_loaded()

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

        st.warning("XGBoost training requires specific formatted data. Use generated data or carefully formatted custom data.")

        # Data source selection
        data_source_xgb = st.radio(
            "Choose training data source:",
            ["Use existing data (Recommended)", "Upload custom data"],
            key="xgb_data_source"
        )

        training_df_xgb = None

        if data_source_xgb == "Use existing data (Recommended)":
            default_path = 'data/xgboost_corrector_training.csv'

            if os.path.exists(default_path):
                st.success(f"Found training data: `{default_path}`")
                try:
                    training_df_xgb = pd.read_csv(default_path)

                    required_cols = ['invalid', 'valid', 'operations']
                    if all(col in training_df_xgb.columns for col in required_cols):
                        with st.expander("Preview Data"):
                            st.dataframe(training_df_xgb.head(20))
                            st.metric("Total Samples", len(training_df_xgb))
                    else:
                        st.error(f"CSV must have columns: {', '.join(required_cols)}")
                        training_df_xgb = None
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            else:
                st.warning("No XGBoost training data found. Please run `python ml/generate_aligned_training_data.py` first.")

        else:  # Upload custom data
            xgb_file = st.file_uploader(
                "Upload Training Data (CSV)",
                type=['csv'],
                key='xgb_upload',
                help="CSV with 'invalid', 'valid', and 'operations' columns"
            )

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
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

        # Train button
        if training_df_xgb is not None:
            if st.button("Train XGBoost Model", type="primary", key='train_xgb'):
                progress_placeholder = st.empty()

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

                        progress_placeholder.info(f"Preparing features from {len(training_data)} samples...")

                        # Prepare features
                        X, y = corrector.prepare_training_data(training_data)

                        progress_placeholder.info("Training XGBoost model...")

                        # Train
                        unique_labels = pd.unique(y)
                        label_map = {old: new for new, old in enumerate(unique_labels)}
                        y_remapped = pd.array([label_map[label] for label in y])

                        corrector.inference_label_map = {new: old for old, new in label_map.items()}

                        corrector.model = xgb.XGBClassifier(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            objective='multi:softmax',
                            num_class=len(unique_labels),
                            random_state=42
                        )

                        corrector.model.fit(X, y_remapped)

                        # Evaluate
                        accuracy = corrector.model.score(X, y)

                        progress_placeholder.info("Saving model...")

                        # Save
                        save_path = 'saved_models/edit_distance_corrector.pkl'
                        corrector.save(save_path)

                        progress_placeholder.empty()

                        st.success("Training completed successfully!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Training Accuracy", f"{accuracy:.1%}")
                        with col2:
                            st.metric("Model Saved", "Yes")

                        st.info(f"Model saved to: `{save_path}`")
                        st.info("Please refresh the page to load the new model")

                        st.balloons()

                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

# Sidebar
with st.sidebar:
    st.header("ML Model Info")

    if st.session_state.ml_model_loaded:
        st.success("Logistic Regression Model Loaded")
        st.info("Phone validation using ML model (+ sign + at least 7 digits rule)")
    else:
        st.warning("ML Model Not Available")
        if hasattr(st.session_state, 'model_load_error') and st.session_state.model_load_error:
            st.error(f"Error: {st.session_state.model_load_error}")
        else:
            st.error("Please train the logistic regression model first")

        # Add training button
        if st.button("Train Model Now", key="train_model_btn"):
            with st.spinner("Training logistic regression model..."):
                try:
                    from ml.model_trainer import train_from_csv_file
                    trainer, results = train_from_csv_file('data/training_data.csv')

                    # Reload the validator
                    st.session_state.ml_phone_validator = PhoneValidator('saved_models/phone_validator_model.pkl')
                    st.session_state.ml_model_loaded = st.session_state.ml_phone_validator.is_model_loaded()
                    st.session_state.model_load_error = None

                    if st.session_state.ml_model_loaded:
                        st.success(f"Model trained successfully! Accuracy: {results['accuracy']:.3f}")
                        st.rerun()  # Refresh the app
                    else:
                        st.error("Training completed but model still not loading")
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    st.header("Supported Data Types")
    st.write("- Phone Numbers (Logistic Regression ML)")
# Removed Blood Sugar validation - focusing on phone validation only
    st.write("- Generic Data (Basic validation)")

# Removed debug section - app working correctly