"""
ML Data Validator - Streamlit App

A clean, focused ML validator:
- Train on YOUR data
- Validate ANY column type
- Suggest corrections
- No complexity, just pure ML

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
from ml.validator import GenericMLValidator
from ml.corrector import GenericMLCorrector
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

st.set_page_config(page_title="ML Validator", layout="wide")

# Hide Streamlit's automatic header anchor links
st.markdown("""
    <style>
        .css-15zrgzn {display: none}
        .css-eczf16 {display: none}
        .css-jn99sy {display: none}
        header .css-18ni7ap {display: none}
        h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
            display: none !important;
        }
        .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================

st.title("Simple ML Data Validator")

# ==================== TABS ====================

tab_validate, tab_train = st.tabs(["Validate Data", "Train Models"])

# ==================== VALIDATION TAB ====================

with tab_validate:
    st.header("Validate Your Data")

    # Initialize session state
    if 'validated_df' not in st.session_state:
        st.session_state.validated_df = None
    if 'corrections_data' not in st.session_state:
        st.session_state.corrections_data = None
    if 'modified_cells' not in st.session_state:
        st.session_state.modified_cells = set()  # Track which cells were modified (row, col)
    if 'cell_validity' not in st.session_state:
        st.session_state.cell_validity = {}  # Track validity per cell
    if 'column_mappings' not in st.session_state:
        st.session_state.column_mappings = {}  # Track which columns are being validated

    # Step 1: Upload CSV
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file:
        # Check if this is a new file upload - clear previous validation results
        current_filename = uploaded_file.name
        if 'current_uploaded_file' not in st.session_state:
            st.session_state.current_uploaded_file = None

        # If a different file is uploaded, clear all validation results
        if st.session_state.current_uploaded_file != current_filename:
            st.session_state.validated_df = None
            st.session_state.corrections_data = None
            st.session_state.modified_cells = set()
            st.session_state.cell_validity = {}
            st.session_state.column_mappings = {}
            st.session_state.current_uploaded_file = current_filename
            # Also clear original_df if it exists
            if 'original_df' in st.session_state:
                st.session_state.original_df = None

        df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded! Shape: {df.shape}")

        # Step 2: Configure column validation
        st.subheader("Configure Validation")

        # List available models
        models_dir = "models"
        available_models = []
        if os.path.exists(models_dir):
            available_models = [f.replace('_validator.pkl', '')
                              for f in os.listdir(models_dir)
                              if f.endswith('_validator.pkl')]

        if not available_models:
            st.warning("No trained models found. Go to 'Train Models' tab first.")
        else:
            # Initialize session state for column mappings
            if 'column_mappings' not in st.session_state:
                st.session_state.column_mappings = {}

            st.write("**Map columns to validators:**")

            # Allow user to select which columns to validate and which validator to use
            column_mappings = {}
            for column in df.columns:
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**{column}**")
                with col2:
                    validator_choice = st.selectbox(
                        f"Validator for {column}",
                        ["(Skip)"] + available_models,
                        key=f"validator_{column}",
                        label_visibility="collapsed"
                    )
                    if validator_choice != "(Skip)":
                        column_mappings[column] = validator_choice
                with col3:
                    if validator_choice != "(Skip)":
                        st.write("")

            st.session_state.column_mappings = column_mappings

            # Step 3: Validate
            if column_mappings and st.button("Validate", type="primary"):
                with st.spinner("Validating..."):
                    # Track validity per cell (row, column) -> is_valid
                    cell_validity = {}
                    all_corrections = {}  # Store corrections per column

                    # Validate each column
                    for column, validator_name in column_mappings.items():
                        validator_path = f"{models_dir}/{validator_name}_validator.pkl"
                        validator = GenericMLValidator(validator_path)

                        if validator.is_trained:
                            # Validate this column
                            results = validator.validate_batch(df[column].astype(str).tolist())

                            # Store validity for each cell
                            for idx, (is_valid, confidence) in enumerate(results):
                                cell_validity[(idx, column)] = is_valid

                            # Get corrections for invalid cells in this column
                            corrector_path = f"{models_dir}/{validator_name}_corrector.pkl"
                            corrector = None
                            if os.path.exists(corrector_path):
                                corrector = GenericMLCorrector(corrector_path)

                            for idx, row in df.iterrows():
                                if not results[idx][0]:  # If this cell is invalid
                                    original = str(row[column])
                                    corrected = None

                                    # Try to get correction if corrector exists
                                    if corrector:
                                        corrected = corrector.correct(original)

                                    # Always add entry for invalid cells
                                    if idx not in all_corrections:
                                        all_corrections[idx] = {}

                                    # Store correction (or original if no correction available)
                                    suggested = corrected if (corrected and corrected != original) else original
                                    has_correction = corrected is not None and corrected != original

                                    all_corrections[idx][column] = {
                                        'original': original,
                                        'suggested': suggested,
                                        'has_correction': has_correction
                                    }

                    # Store in session state
                    st.session_state.validated_df = df
                    st.session_state.original_df = df.copy()
                    st.session_state.cell_validity = cell_validity  # Track which cells are valid/invalid
                    st.session_state.modified_cells = set()  # Track modified cells as (row, col) tuples
                    st.session_state.column_mappings = column_mappings

                    # Format corrections for display
                    corrections = []
                    for idx, col_corrections in all_corrections.items():
                        for column, correction_data in col_corrections.items():
                            corrections.append({
                                'Row Index': idx,
                                'Row': idx + 1,
                                'Column': column,
                                'Original': correction_data['original'],
                                'Suggested': correction_data['suggested'],
                                'Has Correction': correction_data['has_correction']
                            })

                    st.session_state.corrections_data = corrections if corrections else None

        # Show results if validated
        if st.session_state.validated_df is not None:
            df_display = st.session_state.validated_df

            # Calculate stats based on cell-level validity
            if 'cell_validity' in st.session_state and 'column_mappings' in st.session_state:
                total_cells = len(df_display) * len(st.session_state.column_mappings)
                valid_cells = sum(1 for v in st.session_state.cell_validity.values() if v)
                invalid_cells = total_cells - valid_cells
                quality = (valid_cells / total_cells * 100) if total_cells > 0 else 0

                # Show metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Valid Cells", valid_cells)
                col2.metric("Invalid Cells", invalid_cells)
                col3.metric("Quality", f"{quality:.1f}%")

            # Show editable table with cell-level coloring using AgGrid
            st.subheader("Validation Results")

            # Create display dataframe with index column (starting from 1)
            df_for_display = df_display.copy()
            df_for_display.insert(0, 'Index', df_for_display.index + 1)

            # Configure AgGrid
            gb = GridOptionsBuilder.from_dataframe(df_for_display)

            # Make Index column non-editable, narrower, and no menu
            # Add custom cell style to reduce padding
            index_cell_style = JsCode("""
                function(params) {
                    return {
                        'padding': '2px 4px',
                        'textAlign': 'center'
                    };
                }
            """)
            gb.configure_column(
                'Index',
                editable=False,
                width=35,
                minWidth=35,
                maxWidth=50,
                suppressMenu=True,
                filter=False,
                sortable=False,
                cellStyle=index_cell_style
            )

            # Make all data columns editable and add cell-level styling
            for col in df_for_display.columns:
                # Skip Index column (already configured as non-editable)
                if col == 'Index':
                    continue

                gb.configure_column(col, editable=True, suppressMenu=True)

                # Add cell styling for validated columns
                if 'column_mappings' in st.session_state and col in st.session_state.column_mappings:
                    # Prepare data for JavaScript - convert Python tuples to JS arrays and booleans
                    import json

                    # Convert modified_cells set of tuples to list of arrays for JS
                    modified_cells_list = [[int(idx), str(c)] for idx, c in st.session_state.get('modified_cells', set())]

                    # Convert cell_validity dict with proper boolean conversion
                    cell_validity_for_col = {
                        f"{idx}_{col}": ('true' if v else 'false')
                        for (idx, c), v in st.session_state.get('cell_validity', {}).items()
                        if c == col
                    }

                    # Cell-level styling based on validity
                    cell_style = JsCode(f"""
                        function(params) {{
                            const rowIndex = params.node.rowIndex;
                            const colName = "{col}";

                            // Check if cell was modified
                            const modifiedCells = {json.dumps(modified_cells_list)};
                            const isModified = modifiedCells.some(c => c[0] === rowIndex && c[1] === colName);

                            if (isModified) {{
                                return {{
                                    'backgroundColor': '#ff8c00',
                                    'color': '#ffffff'
                                }};
                            }}

                            // Otherwise use validity
                            const cellValidity = {json.dumps(cell_validity_for_col)};
                            const validityKey = rowIndex + '_{col}';
                            const isValid = cellValidity[validityKey];

                            if (isValid === 'true') {{
                                return {{
                                    'backgroundColor': '#28a745',
                                    'color': '#ffffff'
                                }};
                            }} else if (isValid === 'false') {{
                                return {{
                                    'backgroundColor': '#dc3545',
                                    'color': '#ffffff'
                                }};
                            }}

                            return {{}};
                        }}
                    """)
                    gb.configure_column(col, cellStyle=cell_style)

            # Disable filtering and column menu
            gb.configure_default_column(
                filter=False,
                floatingFilter=False,
                suppressMenu=True,
                sortable=False
            )
            gb.configure_grid_options(
                domLayout='normal',
                suppressMenuHide=True,
                enableFilter=False
            )
            grid_options = gb.build()

            # Display AgGrid
            grid_response = AgGrid(
                df_for_display,
                gridOptions=grid_options,
                update_on=['cellValueChanged'],
                allow_unsafe_jscode=True,
                height=400,
                fit_columns_on_grid_load=True,
                theme='streamlit',
                enable_enterprise_modules=False
            )

            # Get edited data
            if grid_response['data'] is not None:
                edited_data = pd.DataFrame(grid_response['data'])

                # Remove Index column from edited data (it's just for display)
                if 'Index' in edited_data.columns:
                    edited_data = edited_data.drop(columns=['Index'])

                # Check for manual edits and track them as modified (cell-level)
                # Compare against original_df to detect first-time edits
                edit_detected = False
                if 'original_df' in st.session_state and 'modified_cells' in st.session_state:
                    for col in df_display.columns:
                        for idx in df_display.index:
                            original_val = st.session_state.original_df.at[idx, col]
                            new_val = edited_data.at[idx, col]
                            cell_key = (idx, col)

                            # Check if both are NaN (NaN should be considered equal)
                            both_nan = pd.isna(original_val) and pd.isna(new_val)

                            # Check if values actually changed
                            if not both_nan:
                                values_changed = str(original_val) != str(new_val)
                                if values_changed and cell_key not in st.session_state.modified_cells:
                                    st.session_state.modified_cells.add(cell_key)
                                    edit_detected = True

                # Update the dataframe with edits
                for col in df_display.columns:
                    df_display[col] = edited_data[col]

                st.session_state.validated_df = df_display

                # Trigger rerun if new edit detected to update styling
                if edit_detected:
                    st.rerun()

            # Show corrections after the table
            if st.session_state.corrections_data:
                # Filter out already applied corrections (cells that are already modified)
                if 'modified_cells' in st.session_state:
                    pending_corrections = [
                        c for c in st.session_state.corrections_data
                        if (c['Row Index'], c['Column']) not in st.session_state.modified_cells
                    ]
                else:
                    pending_corrections = st.session_state.corrections_data

                if pending_corrections:
                    st.subheader("Suggested Corrections")

                    # Sort corrections by row number
                    pending_corrections = sorted(pending_corrections, key=lambda x: x['Row'])

                    # Count corrections that can be applied
                    applicable_corrections = [c for c in pending_corrections if c['Has Correction']]

                    # Add Apply All button if there are applicable corrections
                    if applicable_corrections:
                        if st.button("Apply All Corrections", type="primary"):
                            # Apply all corrections
                            for correction in applicable_corrections:
                                row_idx = correction['Row Index']
                                column = correction['Column']
                                st.session_state.validated_df.at[row_idx, column] = correction['Suggested']
                                st.session_state.modified_cells.add((row_idx, column))
                            st.rerun()

                        st.write(f"**{len(applicable_corrections)} correction(s) available**")
                        st.divider()

                    # Add Apply button for each correction
                    for idx, correction in enumerate(pending_corrections):
                        col1, col2, col3, col4, col5 = st.columns([1, 1.5, 2, 2, 1])
                        with col1:
                            st.write(f"**Row {correction['Row']}**")
                        with col2:
                            st.write(f"*{correction['Column']}*")
                        with col3:
                            st.write(f"{correction['Original']}")
                        with col4:
                            if correction['Has Correction']:
                                st.write(f"→ **{correction['Suggested']}**")
                            else:
                                st.write(f"_No suggestion_")
                        with col5:
                            if correction['Has Correction']:
                                if st.button("Apply", key=f"apply_{idx}"):
                                    # Apply correction to session state dataframe
                                    row_idx = correction['Row Index']
                                    column = correction['Column']
                                    st.session_state.validated_df.at[row_idx, column] = correction['Suggested']
                                    st.session_state.modified_cells.add((row_idx, column))  # Track as modified
                                    st.rerun()
                            else:
                                st.write("")  # Empty placeholder
                else:
                    st.success("All corrections have been applied!")

            # Export
            st.subheader("Export Results")
            # Export without is_valid and confidence columns (so it matches original format)
            export_df = st.session_state.validated_df.drop(columns=['is_valid', 'confidence'], errors='ignore')
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                "Download Results as CSV",
                csv_data,
                file_name=f"validated_{uploaded_file.name if uploaded_file else 'data.csv'}",
                mime="text/csv"
            )

# ==================== TRAINING TAB ====================

with tab_train:
    st.header("Train Your ML Models")

    st.markdown("""
    ### How to Train:
    1. Prepare a CSV file with 2 columns:
       - `text`: Your data (phone numbers, emails, addresses, etc.)
       - `label`: Either "valid" or "invalid"
    2. Upload the CSV
    3. Give your model a name (e.g., "phone", "email", "address")
    4. Click "Train Validator"

    ### Example CSV:
    ```csv
    text,label
    +1234567890,valid
    123,invalid
    +65 9123 4567,valid
    abc123,invalid
    ```
    """)

    # Upload training file
    training_file = st.file_uploader("Upload Training Data CSV", type=['csv'], key="training")

    if training_file:
        train_df = pd.read_csv(training_file)

        # Check columns
        if 'text' in train_df.columns and 'label' in train_df.columns:
            st.success(f"Training data loaded: {len(train_df)} examples")

            # Show sample
            st.subheader("Sample Data")
            st.dataframe(train_df.head(10), use_container_width=True)

            # Show stats
            valid_count = (train_df['label'].str.lower() == 'valid').sum()
            invalid_count = len(train_df) - valid_count

            col1, col2 = st.columns(2)
            col1.metric("Valid Examples", valid_count)
            col2.metric("Invalid Examples", invalid_count)

            # Model name
            model_name = st.text_input("Model Name (e.g., phone, email, address)", value="custom")

            # Train button
            if st.button("Train Validator & Corrector", type="primary"):
                with st.spinner(f"Training {model_name} validator and corrector..."):
                    # Create validator
                    validator = GenericMLValidator()

                    # Train validator
                    training_data = list(zip(train_df['text'], train_df['label']))
                    validator.train(training_data, model_name)

                    # Save validator
                    os.makedirs("models", exist_ok=True)
                    validator_path = f"models/{model_name}_validator.pkl"
                    validator.save(validator_path)

                    # Create and train corrector with valid examples
                    corrector = GenericMLCorrector()

                    # Extract valid examples from training data
                    valid_examples = train_df[train_df['label'].str.lower() == 'valid']['text'].tolist()
                    corrector.train(valid_examples, model_name)

                    # Save corrector
                    corrector_path = f"models/{model_name}_corrector.pkl"
                    corrector.save(corrector_path)

                    st.success(f"Validator trained and saved to {validator_path}!")
                    st.success(f"Corrector saved to {corrector_path}!")

        else:
            st.error("CSV must have columns: 'text' and 'label'")

    # Show available models
    st.subheader("Trained Models")

    models_dir = "models"
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith('_validator.pkl')]

        if models:
            model_list = []
            for model_file in models:
                model_name = model_file.replace('_validator.pkl', '')
                model_path = os.path.join(models_dir, model_file)
                model_list.append({
                    'Name': model_name,
                    'File': model_file,
                    'Size': f"{os.path.getsize(model_path) / 1024:.1f} KB"
                })

            st.dataframe(pd.DataFrame(model_list), use_container_width=True)
        else:
            st.info("No trained models yet. Upload training data above to get started.")
    else:
        st.info("No models directory yet. Train your first model above!")

