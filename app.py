"""
ML Data Validator - Streamlit App

A clean, focused ML validator:
- Train on YOUR data (any CSV, all rows treated as valid)
- Validate ANY column type
- Suggest corrections
- No complexity, just pure ML

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import json
from ml.validator import UnifiedMLValidator
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
            st.dataframe(df.head(10), use_container_width=True)

        st.divider()

        # ========== STEP 2: Select Model ==========
        models_dir = "models"
        available_models = []
        if os.path.exists(models_dir):
            available_models = [f.replace('.pkl', '')
                              for f in os.listdir(models_dir)
                              if f.endswith('.pkl')]

        if not available_models:
            st.warning("No trained models found. Go to 'Train Models' tab first.")
        else:
            st.markdown('<span class="step-badge">Step 2</span> **Select trained model**', unsafe_allow_html=True)

            selected_model = st.selectbox(
                "Choose a model:",
                available_models,
                help="Select a model that was trained on similar data"
            )

            # Load model to show column info
            if selected_model:
                model_path = f"{models_dir}/{selected_model}.pkl"
                preview_validator = UnifiedMLValidator(model_path)

                if preview_validator.is_trained:
                    trained_cols = preview_validator.get_trained_columns()

                    # Auto-match columns
                    column_mapping = preview_validator._match_columns(df.columns.tolist())

                    with st.expander("Column Matching", expanded=True):
                        st.markdown(f"**Model trained on:** {', '.join(trained_cols)}")

                        if column_mapping:
                            st.markdown("**Auto-detected matches:**")
                            for input_col, trained_col in column_mapping.items():
                                st.markdown(f"  - `{input_col}` → `{trained_col}`")

                        unmatched = [c for c in df.columns if c not in column_mapping]
                        if unmatched:
                            st.markdown(f"**Unmatched columns (will be skipped):** {', '.join(unmatched)}")

                    st.session_state.column_mappings = column_mapping

            st.divider()

            # ========== STEP 3: Validate ==========
            st.markdown('<span class="step-badge">Step 3</span> **Run validation**', unsafe_allow_html=True)

            if selected_model and st.button("Validate", type="primary", use_container_width=True):
                # Progress bar
                progress_bar = st.progress(0, text="Loading model...")

                # Load validator
                model_path = f"{models_dir}/{selected_model}.pkl"
                validator = UnifiedMLValidator(model_path)

                if not validator.is_trained:
                    st.error("Failed to load model")
                else:
                    column_mapping = validator._match_columns(df.columns.tolist())

                    cell_validity = {}
                    all_corrections = {}
                    total_columns = len(column_mapping)

                    for col_idx, (input_col, trained_col) in enumerate(column_mapping.items()):
                        progress_bar.progress(
                            (col_idx) / total_columns,
                            text=f"Validating column: {input_col}..."
                        )

                        # Validate batch
                        results = validator.validate_batch(df[input_col].astype(str).tolist(), trained_col)

                        for idx, (is_valid, confidence) in enumerate(results):
                            cell_validity[(idx, input_col)] = is_valid

                        # Get corrections for invalid cells
                        for idx, row in df.iterrows():
                            if not results[idx][0]:
                                original = str(row[input_col])
                                corrected = validator.correct(original, trained_col)
                                reason = validator.explain_invalidity(original, trained_col)

                                if idx not in all_corrections:
                                    all_corrections[idx] = {}

                                suggested = corrected if (corrected and corrected != original) else original
                                has_correction = corrected is not None and corrected != original

                                all_corrections[idx][input_col] = {
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
                    st.session_state.column_mappings = column_mapping

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
        **New Simplified Training:**

        1. Upload ANY CSV file (your actual data!)
        2. All rows are treated as **valid examples**
        3. Give your model a name
        4. Optionally exclude columns (like ID, timestamp)
        5. Click "Train Model"

        The system automatically:
        - Trains a validator for each column
        - Generates synthetic invalid examples
        - Stores everything in one model file

        **Example:** Upload a CSV with columns `name`, `email`, `phone` - the model learns what valid names, emails, and phones look like from your data.
        """)

    # Upload training file
    st.markdown('<span class="step-badge">Step 1</span> **Upload your data CSV**', unsafe_allow_html=True)
    st.caption("All rows will be treated as valid examples")

    training_file = st.file_uploader("Upload Training Data CSV", type=['csv'], key="training", label_visibility="collapsed")

    if training_file:
        train_df = pd.read_csv(training_file)

        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(train_df))
        col2.metric("Columns", len(train_df.columns))
        col3.metric("File", training_file.name)

        # Preview
        with st.expander("Preview training data", expanded=True):
            st.dataframe(train_df.head(10), use_container_width=True)

        st.divider()

        # Column selection
        st.markdown('<span class="step-badge">Step 2</span> **Configure columns**', unsafe_allow_html=True)

        all_columns = train_df.columns.tolist()
        exclude_columns = st.multiselect(
            "Exclude columns (optional):",
            all_columns,
            help="Skip columns like ID, timestamp, etc. that shouldn't be validated"
        )

        columns_to_train = [c for c in all_columns if c not in exclude_columns]
        st.info(f"Will train on {len(columns_to_train)} columns: {', '.join(columns_to_train)}")

        st.divider()

        # Model name and train
        st.markdown('<span class="step-badge">Step 3</span> **Choose training mode**', unsafe_allow_html=True)

        # Check for existing models
        existing_models = []
        if os.path.exists("models"):
            existing_models = [f.replace('.pkl', '') for f in os.listdir("models") if f.endswith('.pkl')]

        training_mode = st.radio(
            "Training mode:",
            ["Create new model", "Add data to existing model"],
            help="Create new: Start fresh. Add data: Improve existing model with more examples."
        )

        if training_mode == "Create new model":
            model_name = st.text_input("Model name", value="my_model", placeholder="e.g., customer_data, sales_records")
        else:
            if not existing_models:
                st.warning("No existing models found. Please create a new model first.")
                model_name = None
            else:
                model_name = st.selectbox("Select model to improve:", existing_models)

        # Reference lists option
        st.markdown("**Reference Lists (Optional):**")
        use_reference_lists = st.checkbox(
            "Load reference lists for standard fields",
            value=True,
            help="Automatically load valid values for countries, etc. from reference_lists/ folder"
        )

        button_label = "Train Model" if training_mode == "Create new model" else "Add Data & Retrain"
        if st.button(button_label, type="primary", use_container_width=True):
            if not model_name:
                st.error("Please select or enter a model name.")
            elif len(columns_to_train) == 0:
                st.error("No columns selected for training. Please include at least one column.")
            else:
                progress = st.progress(0, text=f"Training {model_name}...")

                if training_mode == "Create new model":
                    # Create fresh model
                    validator = UnifiedMLValidator()
                    progress.progress(10, text="Initializing...")
                    metrics = validator.train(train_df, model_name, exclude_columns)
                else:
                    # Add to existing model
                    model_path = f"models/{model_name}.pkl"
                    validator = UnifiedMLValidator(model_path)
                    progress.progress(10, text="Loading existing model...")

                    if not validator.is_trained:
                        st.error(f"Failed to load model: {model_name}")
                        st.stop()

                    progress.progress(30, text="Adding new data...")
                    metrics = validator.add_training_data(train_df, retrain=True)

                # Load reference lists if enabled
                if use_reference_lists and os.path.exists("reference_lists"):
                    progress.progress(70, text="Loading reference lists...")
                    ref_results = validator.load_reference_lists_from_dir("reference_lists")
                    if ref_results:
                        st.info(f"Loaded reference lists: {', '.join(f'{k} ({v} values)' for k, v in ref_results.items())}")

                progress.progress(80, text="Saving model...")
                os.makedirs("models", exist_ok=True)
                model_path = f"models/{model_name}.pkl"
                validator.save(model_path)

                progress.progress(100, text="Done!")
                if training_mode == "Create new model":
                    st.success(f"Model **{model_name}** trained and saved!")
                else:
                    st.success(f"Model **{model_name}** updated with new data!")

                # Display training metrics
                st.divider()
                st.subheader("Training Metrics")

                # Per-column metrics
                for col_name, col_metrics in metrics.items():
                    with st.expander(f"Column: {col_name}", expanded=True):
                        # Basic stats
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        stat_col1.metric("Unique Valid Examples", col_metrics.get('unique_valid', 'N/A'))
                        stat_col2.metric("Total Samples", col_metrics.get('total_samples', 'N/A'))

                        if col_metrics.get('used_split', False):
                            stat_col3.metric("Test Size", col_metrics.get('test_size', 'N/A'))

                            # Train vs Test comparison
                            train_col, test_col = st.columns(2)

                            with train_col:
                                st.markdown("**Train Set**")
                                m1, m2 = st.columns(2)
                                m1.metric("Accuracy", f"{col_metrics.get('train_accuracy', 0):.1%}")
                                m2.metric("F1 Score", f"{col_metrics.get('train_f1', 0):.1%}")

                            with test_col:
                                st.markdown("**Test Set**")
                                m1, m2 = st.columns(2)
                                m1.metric("Accuracy", f"{col_metrics.get('test_accuracy', 0):.1%}")
                                m2.metric("F1 Score", f"{col_metrics.get('test_f1', 0):.1%}")

                            if 'test_confusion_matrix' in col_metrics:
                                display_confusion_matrix(col_metrics['test_confusion_matrix'], "Test Confusion Matrix")
                        else:
                            stat_col3.metric("Split", "N/A (small dataset)")
                            st.warning("Dataset too small for train/test split. Metrics may be overfit.")

                            m1, m2 = st.columns(2)
                            m1.metric("Accuracy", f"{col_metrics.get('train_accuracy', 0):.1%}")
                            m2.metric("F1 Score", f"{col_metrics.get('train_f1', 0):.1%}")

    st.divider()

    # Show available models (collapsible)
    with st.expander("Trained Models", expanded=True):
        models_dir = "models"
        if os.path.exists(models_dir):
            # Look for new unified models
            models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]

            if models:
                model_list = []
                for model_file in models:
                    model_name = model_file.replace('.pkl', '')
                    model_path = os.path.join(models_dir, model_file)

                    # Load to get column info
                    try:
                        temp_validator = UnifiedMLValidator(model_path)
                        columns = temp_validator.get_trained_columns()
                        columns_str = ', '.join(columns[:5])
                        if len(columns) > 5:
                            columns_str += f" (+{len(columns)-5} more)"
                    except Exception:
                        columns_str = "Unknown"

                    model_list.append({
                        'Name': model_name,
                        'Size': f"{os.path.getsize(model_path) / 1024:.1f} KB",
                        'Columns': columns_str,
                    })

                st.dataframe(pd.DataFrame(model_list), use_container_width=True, hide_index=True)
            else:
                st.info("No trained models yet. Upload training data above to get started.")
        else:
            st.info("No models directory yet. Train your first model above!")
