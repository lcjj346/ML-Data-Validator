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
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

st.set_page_config(page_title="ML Data Validator", layout="wide", page_icon="🔍")


def safe_cast_value(value, target_dtype):
    """Cast a value to the target dtype, handling type mismatches gracefully."""
    try:
        if pd.api.types.is_float_dtype(target_dtype):
            return float(value)
        elif pd.api.types.is_integer_dtype(target_dtype):
            return int(float(value))
        elif pd.api.types.is_bool_dtype(target_dtype):
            return bool(value)
        else:
            return value
    except (ValueError, TypeError):
        return value


def display_confusion_matrix(cm, title="Confusion Matrix"):
    """Display confusion matrix with labels."""
    cm_df = pd.DataFrame(
        [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]],
        index=['Actual Invalid', 'Actual Valid'],
        columns=['Predicted Invalid', 'Predicted Valid']
    )
    st.markdown(f"**{title}**")
    st.dataframe(cm_df.style.background_gradient(cmap='Blues', axis=None))


# Custom CSS for better UI/UX
st.markdown("""
    <style>
        /* Hide Streamlit's automatic header anchor links */
        h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
            display: none !important;
        }
        .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a {
            display: none !important;
        }

        /* Smooth scroll */
        html { scroll-behavior: smooth; }

        /* Button transitions */
        .stButton > button {
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            transform: scale(1.03);
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
        }

        /* Step badges */
        .step-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-right: 8px;
        }

        /* Correction card */
        .correction-card {
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 4px;
            border-left: 4px solid;
            transition: all 0.2s ease;
        }
        .correction-card:hover {
            transform: translateX(4px);
        }
        .correction-has-fix {
            border-left-color: #51cf66;
            background-color: rgba(81, 207, 102, 0.05);
        }
        .correction-no-fix {
            border-left-color: #ff6b6b;
            background-color: rgba(255, 107, 107, 0.05);
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
        }

        /* File uploader compact */
        [data-testid="stFileUploader"] {
            padding-bottom: 0;
        }

        /* Quality badge */
        .quality-high { color: #51cf66; }
        .quality-medium { color: #fcc419; }
        .quality-low { color: #ff6b6b; }
    </style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================

st.title("ML Data Validator")
st.caption("Train on your data, validate any column type, suggest corrections.")

# ==================== TABS ====================

tab_validate, tab_train = st.tabs(["Validate Data", "Train Models"])

# ==================== VALIDATION TAB ====================

with tab_validate:

    # Initialize session state
    if 'validated_df' not in st.session_state:
        st.session_state.validated_df = None
    if 'corrections_data' not in st.session_state:
        st.session_state.corrections_data = None
    if 'modified_cells' not in st.session_state:
        st.session_state.modified_cells = set()
    if 'cell_validity' not in st.session_state:
        st.session_state.cell_validity = {}
    if 'column_mappings' not in st.session_state:
        st.session_state.column_mappings = {}
    if 'last_correction' not in st.session_state:
        st.session_state.last_correction = None

    # Show success toast if correction was just applied
    if st.session_state.last_correction:
        correction_info = st.session_state.last_correction
        st.toast(f"Applied: {correction_info['original']} → {correction_info['suggested']}", icon="✅")
        st.markdown('<script>window.scrollTo({top: 0, behavior: "smooth"});</script>', unsafe_allow_html=True)
        st.session_state.last_correction = None

    # ========== STEP 1: Upload ==========
    st.markdown('<span class="step-badge">Step 1</span> **Upload your CSV file**', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], label_visibility="collapsed")

    if uploaded_file:
        # Check if this is a new file upload - clear previous validation results
        current_filename = uploaded_file.name
        if 'current_uploaded_file' not in st.session_state:
            st.session_state.current_uploaded_file = None

        if st.session_state.current_uploaded_file != current_filename:
            st.session_state.validated_df = None
            st.session_state.corrections_data = None
            st.session_state.modified_cells = set()
            st.session_state.cell_validity = {}
            st.session_state.column_mappings = {}
            st.session_state.current_uploaded_file = current_filename
            if 'original_df' in st.session_state:
                st.session_state.original_df = None

        df = pd.read_csv(uploaded_file)

        # File info in columns
        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("Rows", df.shape[0])
        info_col2.metric("Columns", df.shape[1])
        info_col3.metric("File", uploaded_file.name)

        # Preview data
        with st.expander("Preview uploaded data", expanded=False):
            st.dataframe(df.head(10), width='stretch')

        st.divider()

        # ========== STEP 2: Configure ==========
        models_dir = "models"
        available_models = []
        if os.path.exists(models_dir):
            available_models = [f.replace('_validator.pkl', '')
                              for f in os.listdir(models_dir)
                              if f.endswith('_validator.pkl')]

        if not available_models:
            st.warning("No trained models found. Go to 'Train Models' tab first.")
        else:
            st.markdown('<span class="step-badge">Step 2</span> **Map columns to validators**', unsafe_allow_html=True)

            # Define base validation types
            base_validators = ['name', 'email', 'phone', 'country', 'blood_sugar', 'age', 'address']

            # Auto-map columns
            column_mappings = {}
            unmapped_columns = []

            for column in df.columns:
                column_lower = column.lower().replace('_', '').replace('-', '').replace(' ', '')
                matched = False
                for base_val in base_validators:
                    base_val_normalized = base_val.replace('_', '').replace('-', '').replace(' ', '')
                    if base_val_normalized in column_lower:
                        if base_val in available_models:
                            column_mappings[column] = base_val
                            matched = True
                            break
                        elif base_val_normalized in available_models:
                            column_mappings[column] = base_val_normalized
                            matched = True
                            break
                if not matched:
                    unmapped_columns.append(column)

            with st.expander("Column Mapping Configuration", expanded=bool(unmapped_columns)):
                # Show auto-mapped
                if column_mappings:
                    st.markdown("**Auto-detected mappings:**")
                    for column, validator in column_mappings.items():
                        col1, col2 = st.columns([2, 2])
                        with col1:
                            st.write(f"**{column}**")
                        with col2:
                            st.write(f"✓ {validator}")

                # Show unmapped
                if unmapped_columns:
                    st.markdown("---")
                    st.markdown("**Map additional columns:**")
                    for column in unmapped_columns:
                        col1, col2 = st.columns([2, 2])
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
                else:
                    st.success("All columns auto-mapped!")

            st.session_state.column_mappings = column_mappings

            st.divider()

            # ========== STEP 3: Validate ==========
            st.markdown('<span class="step-badge">Step 3</span> **Run validation**', unsafe_allow_html=True)

            if column_mappings and st.button("Validate", type="primary", use_container_width=True):
                # Progress bar
                progress_bar = st.progress(0, text="Starting validation...")

                cell_validity = {}
                all_corrections = {}
                total_columns = len(column_mappings)

                for col_idx, (column, validator_name) in enumerate(column_mappings.items()):
                    progress_bar.progress(
                        (col_idx) / total_columns,
                        text=f"Validating column: {column}..."
                    )

                    validator_path = f"{models_dir}/{validator_name}_validator.pkl"
                    try:
                        validator = GenericMLValidator(validator_path)
                    except Exception as e:
                        st.warning(f"Failed to load validator for '{column}': {e}")
                        continue

                    if validator.is_trained:
                        results = validator.validate_batch(df[column].astype(str).tolist())

                        for idx, (is_valid, confidence) in enumerate(results):
                            cell_validity[(idx, column)] = is_valid

                        corrector_path = f"{models_dir}/{validator_name}_corrector.pkl"
                        corrector = None
                        if os.path.exists(corrector_path):
                            try:
                                corrector = GenericMLCorrector(corrector_path)
                            except Exception as e:
                                st.warning(f"Failed to load corrector for '{column}': {e}")

                        for idx, row in df.iterrows():
                            if not results[idx][0]:
                                original = str(row[column])
                                corrected = None

                                if corrector:
                                    corrected = corrector.correct(original)

                                reason = validator.explain_invalidity(original)

                                if idx not in all_corrections:
                                    all_corrections[idx] = {}

                                suggested = corrected if (corrected and corrected != original) else original
                                has_correction = corrected is not None and corrected != original

                                all_corrections[idx][column] = {
                                    'original': original,
                                    'suggested': suggested,
                                    'has_correction': has_correction,
                                    'reason': reason
                                }

                progress_bar.progress(1.0, text="Validation complete!")

                # Store in session state
                st.session_state.validated_df = df
                st.session_state.original_df = df.copy()
                st.session_state.cell_validity = cell_validity
                st.session_state.modified_cells = set()
                st.session_state.column_mappings = column_mappings

                corrections = []
                for idx, col_corrections in all_corrections.items():
                    for column, correction_data in col_corrections.items():
                        corrections.append({
                            'Row Index': idx,
                            'Row': idx + 1,
                            'Column': column,
                            'Original': correction_data['original'],
                            'Suggested': correction_data['suggested'],
                            'Has Correction': correction_data['has_correction'],
                            'Reason': correction_data.get('reason', 'Unknown')
                        })

                st.session_state.corrections_data = corrections if corrections else None
                st.rerun()

        # ==================== RESULTS ====================
        if st.session_state.validated_df is not None:
            df_display = st.session_state.validated_df

            st.divider()

            # ===== Quality Metrics =====
            if 'cell_validity' in st.session_state and 'column_mappings' in st.session_state:
                total_cells = len(df_display) * len(st.session_state.column_mappings)
                valid_cells = sum(1 for v in st.session_state.cell_validity.values() if v)
                invalid_cells = total_cells - valid_cells
                quality = (valid_cells / total_cells * 100) if total_cells > 0 else 0

                # Quality color
                if quality >= 80:
                    quality_class = "quality-high"
                elif quality >= 50:
                    quality_class = "quality-medium"
                else:
                    quality_class = "quality-low"

                col1, col2, col3 = st.columns(3)
                col1.metric("Valid Cells", valid_cells, delta=f"{valid_cells} ok")
                col2.metric("Invalid Cells", invalid_cells, delta=f"-{invalid_cells}" if invalid_cells else "0")
                col3.metric("Quality", f"{quality:.1f}%")

            # ===== Validation Results Table (Collapsible) =====
            with st.expander("Validation Results Table", expanded=True):

                # Create display dataframe with index column
                df_for_display = df_display.copy()
                df_for_display.insert(0, 'Index', df_for_display.index + 1)

                # Configure AgGrid
                gb = GridOptionsBuilder.from_dataframe(df_for_display)

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

                for col in df_for_display.columns:
                    if col == 'Index':
                        continue

                    gb.configure_column(col, editable=True, suppressMenu=True)

                    if 'column_mappings' in st.session_state and col in st.session_state.column_mappings:
                        import json

                        modified_cells_list = [[int(idx), str(c)] for idx, c in st.session_state.get('modified_cells', set())]
                        cell_validity_for_col = {
                            f"{idx}_{col}": ('true' if v else 'false')
                            for (idx, c), v in st.session_state.get('cell_validity', {}).items()
                            if c == col
                        }

                        cell_style = JsCode(f"""
                            function(params) {{
                                const rowIndex = params.node.rowIndex;
                                const colName = "{col}";
                                const modifiedCells = {json.dumps(modified_cells_list)};
                                const isModified = modifiedCells.some(c => c[0] === rowIndex && c[1] === colName);

                                if (isModified) {{
                                    return {{
                                        'backgroundColor': '#ff8c00',
                                        'color': '#ffffff'
                                    }};
                                }}

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

                    if 'Index' in edited_data.columns:
                        edited_data = edited_data.drop(columns=['Index'])

                    edited_data.index = df_display.index

                    edit_detected = False
                    if 'original_df' in st.session_state and 'modified_cells' in st.session_state:
                        for col in df_display.columns:
                            for idx in df_display.index:
                                original_val = st.session_state.original_df.at[idx, col]
                                new_val = edited_data.at[idx, col]
                                cell_key = (idx, col)

                                both_nan = pd.isna(original_val) and pd.isna(new_val)

                                if not both_nan:
                                    values_changed = str(original_val) != str(new_val)
                                    if values_changed and cell_key not in st.session_state.modified_cells:
                                        st.session_state.modified_cells.add(cell_key)
                                        edit_detected = True

                    for col in df_display.columns:
                        df_display[col] = edited_data[col]

                    st.session_state.validated_df = df_display

                    if edit_detected:
                        st.rerun()

                # Color legend
                st.markdown(
                    '&nbsp;&nbsp; 🟩 Valid &nbsp;&nbsp; 🟥 Invalid &nbsp;&nbsp; 🟧 Manually edited',
                    unsafe_allow_html=True
                )

            # ===== Corrections Section (Collapsible) =====
            if st.session_state.corrections_data:
                if 'modified_cells' in st.session_state:
                    pending_corrections = [
                        c for c in st.session_state.corrections_data
                        if (c['Row Index'], c['Column']) not in st.session_state.modified_cells
                    ]
                else:
                    pending_corrections = st.session_state.corrections_data

                if pending_corrections:
                    applicable_corrections = [c for c in pending_corrections if c['Has Correction']]
                    no_suggestion = [c for c in pending_corrections if not c['Has Correction']]

                    with st.expander(
                        f"Suggested Corrections ({len(applicable_corrections)} fixable, {len(no_suggestion)} need review)",
                        expanded=True
                    ):
                        # Filter and Apply All controls
                        unique_columns = sorted(list(set([c['Column'] for c in pending_corrections])))

                        filter_col, btn_col = st.columns([2, 2])
                        with filter_col:
                            selected_column = st.selectbox(
                                "Filter by column:",
                                ["All Columns"] + unique_columns,
                                key="correction_filter"
                            )
                        with btn_col:
                            if applicable_corrections:
                                st.write("")  # Spacer for alignment
                                if st.button(
                                    f"Apply All {len(applicable_corrections)} Corrections",
                                    type="primary",
                                    use_container_width=True
                                ):
                                    for correction in applicable_corrections:
                                        row_idx = correction['Row Index']
                                        column = correction['Column']
                                        target_dtype = st.session_state.validated_df[column].dtype
                                        casted_value = safe_cast_value(correction['Suggested'], target_dtype)
                                        st.session_state.validated_df.at[row_idx, column] = casted_value
                                        st.session_state.modified_cells.add((row_idx, column))
                                    st.session_state.last_correction = {
                                        'original': f'{len(applicable_corrections)} cells',
                                        'suggested': 'corrected'
                                    }
                                    st.rerun()

                        # Filter
                        if selected_column != "All Columns":
                            pending_corrections = [c for c in pending_corrections if c['Column'] == selected_column]

                        pending_corrections = sorted(pending_corrections, key=lambda x: x['Row'])

                        st.divider()

                        # Column headers
                        hdr1, hdr2, hdr3, hdr4, hdr5, hdr6 = st.columns([0.6, 1.0, 1.5, 1.5, 2, 0.8])
                        with hdr1:
                            st.markdown("**Row**")
                        with hdr2:
                            st.markdown("**Column**")
                        with hdr3:
                            st.markdown("**Original**")
                        with hdr4:
                            st.markdown("**Suggestion**")
                        with hdr5:
                            st.markdown("**Reason**")
                        with hdr6:
                            st.markdown("**Action**")
                        st.markdown("<hr style='margin: 0 0 8px 0; border: none; border-top: 1px solid rgba(255,255,255,0.15);'>", unsafe_allow_html=True)

                        # Correction rows
                        for idx, correction in enumerate(pending_corrections):
                            has_fix = correction['Has Correction']

                            col1, col2, col3, col4, col5, col6 = st.columns([0.6, 1.0, 1.5, 1.5, 2, 0.8])
                            with col1:
                                st.markdown(f"**{correction['Row']}**")
                            with col2:
                                st.markdown(f"`{correction['Column']}`")
                            with col3:
                                st.markdown(
                                    f"<span style='color: #ff6b6b; font-weight: 500;'>{correction['Original']}</span>",
                                    unsafe_allow_html=True
                                )
                            with col4:
                                if has_fix:
                                    st.markdown(
                                        f"<span style='color: #51cf66;'>→ <b>{correction['Suggested']}</b></span>",
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown("<span style='color: #868e96;'>No suggestion</span>", unsafe_allow_html=True)
                            with col5:
                                reason = correction.get('Reason', 'Unknown')
                                st.markdown(
                                    f"<span style='color: #adb5bd; font-size: 0.9em;'>{reason}</span>",
                                    unsafe_allow_html=True
                                )
                            with col6:
                                if has_fix:
                                    if st.button("Fix", key=f"apply_{idx}", type="primary"):
                                        row_idx = correction['Row Index']
                                        column = correction['Column']
                                        target_dtype = st.session_state.validated_df[column].dtype
                                        casted_value = safe_cast_value(correction['Suggested'], target_dtype)
                                        st.session_state.validated_df.at[row_idx, column] = casted_value
                                        st.session_state.modified_cells.add((row_idx, column))
                                        st.session_state.last_correction = {
                                            'original': correction['Original'],
                                            'suggested': correction['Suggested']
                                        }
                                        st.rerun()

                            # Subtle separator between rows
                            if idx < len(pending_corrections) - 1:
                                st.markdown(
                                    "<hr style='margin: 2px 0; border: none; border-top: 1px solid rgba(255,255,255,0.05);'>",
                                    unsafe_allow_html=True
                                )

                else:
                    st.success("All corrections have been applied! Your data is clean.")
                    st.balloons()

            # ===== Export Section (Collapsible) =====
            with st.expander("Export Results", expanded=True):
                export_df = st.session_state.validated_df.drop(columns=['is_valid', 'confidence'], errors='ignore')
                csv_data = export_df.to_csv(index=False)

                export_col1, export_col2 = st.columns([2, 2])
                with export_col1:
                    st.download_button(
                        "Download Validated CSV",
                        csv_data,
                        file_name=f"validated_{uploaded_file.name if uploaded_file else 'data.csv'}",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
                with export_col2:
                    st.metric("Total Rows", len(export_df))

# ==================== TRAINING TAB ====================

with tab_train:
    st.header("Train Your ML Models")

    # Instructions (collapsible)
    with st.expander("How to Train", expanded=False):
        st.markdown("""
        1. Prepare a CSV file with 2 columns:
           - `text`: Your data (phone numbers, emails, addresses, etc.)
           - `label`: Either "valid" or "invalid"
        2. Upload the CSV below
        3. Give your model a name (e.g., "phone", "email", "address")
        4. Click "Train Validator & Corrector"

        **Important:** Avoid commas in the `text` column as they break CSV structure.
        Use spaces instead (e.g., "123 Main Street New York" not "123 Main Street, New York").

        **Example CSV:**
        ```csv
        text,label
        +1234567890,valid
        123,invalid
        +65 9123 4567,valid
        abc123,invalid
        ```
        """)

    # Upload training file
    st.markdown('<span class="step-badge">Step 1</span> **Upload training data**', unsafe_allow_html=True)
    training_file = st.file_uploader("Upload Training Data CSV", type=['csv'], key="training", label_visibility="collapsed")

    if training_file:
        train_df = pd.read_csv(training_file)

        if 'text' in train_df.columns and 'label' in train_df.columns:

            # Stats
            valid_count = (train_df['label'].str.lower() == 'valid').sum()
            invalid_count = len(train_df) - valid_count

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Examples", len(train_df))
            col2.metric("Valid", valid_count)
            col3.metric("Invalid", invalid_count)

            # Preview
            with st.expander("Preview training data", expanded=False):
                st.dataframe(train_df.head(10), width='stretch')

            st.divider()

            # Model name and train
            st.markdown('<span class="step-badge">Step 2</span> **Name and train your model**', unsafe_allow_html=True)
            model_name = st.text_input("Model name", value="custom", placeholder="e.g., phone, email, address")

            if st.button("Train Validator & Corrector", type="primary", use_container_width=True):
                progress = st.progress(0, text=f"Training {model_name}...")

                # Train validator
                validator = GenericMLValidator()
                training_data = list(zip(train_df['text'], train_df['label']))
                progress.progress(20, text="Extracting features...")
                metrics = validator.train(training_data, model_name)

                progress.progress(50, text="Saving validator...")
                os.makedirs("models", exist_ok=True)
                validator_path = f"models/{model_name}_validator.pkl"
                validator.save(validator_path)

                # Train corrector
                progress.progress(60, text="Training corrector...")
                corrector = GenericMLCorrector()
                valid_examples = train_df[train_df['label'].str.lower() == 'valid']['text'].tolist()
                corrector.train(valid_examples, model_name)

                progress.progress(80, text="Saving corrector...")
                corrector_path = f"models/{model_name}_corrector.pkl"
                corrector.save(corrector_path)

                progress.progress(100, text="Done!")
                st.success(f"Model **{model_name}** trained and saved!")

                # Display training metrics
                st.divider()
                st.subheader("Training Metrics")

                if metrics.get('small_dataset_warning'):
                    st.warning("Dataset too small for train/test split. Metrics shown are from training data (may be overfit).")
                    st.markdown(f"**Total samples:** {metrics.get('total_samples', 'N/A')}")

                    # Show training metrics only
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    metric_col1.metric("Accuracy", f"{metrics.get('train_accuracy', 0):.1%}")
                    metric_col2.metric("Precision", f"{metrics.get('train_precision', 0):.1%}")
                    metric_col3.metric("Recall", f"{metrics.get('train_recall', 0):.1%}")
                    metric_col4.metric("F1 Score", f"{metrics.get('train_f1', 0):.1%}")

                    if 'train_confusion_matrix' in metrics:
                        display_confusion_matrix(metrics['train_confusion_matrix'], "Training Confusion Matrix")

                else:
                    # Show train/test split info
                    st.markdown(f"**Split:** {metrics.get('train_size', 0)} train / {metrics.get('test_size', 0)} test (80/20)")

                    # Two columns for train vs test metrics
                    train_col, test_col = st.columns(2)

                    with train_col:
                        st.markdown("#### Train Set")
                        m1, m2 = st.columns(2)
                        m1.metric("Accuracy", f"{metrics.get('train_accuracy', 0):.1%}")
                        m2.metric("Precision", f"{metrics.get('train_precision', 0):.1%}")
                        m3, m4 = st.columns(2)
                        m3.metric("Recall", f"{metrics.get('train_recall', 0):.1%}")
                        m4.metric("F1 Score", f"{metrics.get('train_f1', 0):.1%}")

                        if 'train_confusion_matrix' in metrics:
                            display_confusion_matrix(metrics['train_confusion_matrix'], "Train Confusion Matrix")

                    with test_col:
                        st.markdown("#### Test Set")
                        m1, m2 = st.columns(2)
                        m1.metric("Accuracy", f"{metrics.get('test_accuracy', 0):.1%}")
                        m2.metric("Precision", f"{metrics.get('test_precision', 0):.1%}")
                        m3, m4 = st.columns(2)
                        m3.metric("Recall", f"{metrics.get('test_recall', 0):.1%}")
                        m4.metric("F1 Score", f"{metrics.get('test_f1', 0):.1%}")

                        if 'test_confusion_matrix' in metrics:
                            display_confusion_matrix(metrics['test_confusion_matrix'], "Test Confusion Matrix")

        else:
            st.error("CSV must have columns: `text` and `label`")

    st.divider()

    # Show available models (collapsible)
    with st.expander("Trained Models", expanded=True):
        models_dir = "models"
        if os.path.exists(models_dir):
            models = [f for f in os.listdir(models_dir) if f.endswith('_validator.pkl')]

            if models:
                model_list = []
                for model_file in models:
                    model_name = model_file.replace('_validator.pkl', '')
                    model_path = os.path.join(models_dir, model_file)
                    corrector_path = os.path.join(models_dir, f"{model_name}_corrector.pkl")
                    has_corrector = os.path.exists(corrector_path)
                    model_list.append({
                        'Name': model_name,
                        'Validator': f"{os.path.getsize(model_path) / 1024:.1f} KB",
                        'Corrector': f"{os.path.getsize(corrector_path) / 1024:.1f} KB" if has_corrector else "N/A",
                    })

                st.dataframe(pd.DataFrame(model_list), width='stretch', hide_index=True)
            else:
                st.info("No trained models yet. Upload training data above to get started.")
        else:
            st.info("No models directory yet. Train your first model above!")
