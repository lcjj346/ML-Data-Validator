import streamlit as st
import pandas as pd
import numpy as np
import os
from ml.validator_model import MLValidator
from ml.model_trainer import ModelTrainer
from ml.data_corrector import DataCorrector
from utils.phone_validator import is_phone_number_valid
from utils.blood_sugar_validator import is_blood_sugar_valid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
import io

# Page configuration
st.set_page_config(
    page_title="ML Data Validator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 ML Data Validator</h1>
    <p>Upload survey data, detect issues with AI, and get intelligent correction suggestions</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'validator' not in st.session_state:
    st.session_state.validator = MLValidator()
    st.session_state.trainer = ModelTrainer()
    st.session_state.corrector = DataCorrector()
    st.session_state.models_trained = False
    st.session_state.current_df = None
    st.session_state.validated_df = None

# Sidebar for model management
with st.sidebar:
    st.header("🤖 Model Management")
    
    # Load existing models
    if st.button("Load Pre-trained Models"):
        if st.session_state.validator.load_models('ml/trained_models/validator_models.pkl'):
            st.success("Models loaded successfully!")
            st.session_state.models_trained = True
        else:
            st.warning("No pre-trained models found")
    
    # Train new models
    if st.button("Train New Models"):
        with st.spinner("Training models..."):
            results = st.session_state.trainer.train_from_sample_data()
            
            if 'error' in results:
                st.error(results['error'])
            else:
                st.success("Models trained successfully!")
                st.session_state.models_trained = True
                
                # Show training results
                for model_name, metrics in results.items():
                    if isinstance(metrics, dict) and 'accuracy' in metrics:
                        st.metric(f"{model_name} Accuracy", f"{metrics['accuracy']:.3f}")

# Main content
st.header("📁 Data Upload & Validation")

uploaded_file = st.file_uploader(
    "Upload your survey data",
    type=["csv", "xlsx"],
    help="Upload CSV or Excel files containing survey data"
)

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"File uploaded successfully! Shape: {df.shape}")
        
        # Store original data
        st.session_state.current_df = df.copy()
        
        # Show original data preview
        with st.expander("📊 Original Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Validation section
        st.header("🔍 Data Validation & Editing")
        
        # Create validation columns
        validated_df = df.copy()
        
        # Apply validation rules
        validation_summary = {}
        
        for column in df.columns:
            if column == 'PhoneNumber':
                validated_df[f"{column}_Valid"] = df[column].apply(is_phone_number_valid)
                invalid_count = (~validated_df[f"{column}_Valid"]).sum()
                validation_summary[column] = invalid_count
            elif column == 'BloodSugar':
                validated_df[f"{column}_Valid"] = df[column].apply(is_blood_sugar_valid)
                invalid_count = (~validated_df[f"{column}_Valid"]).sum()
                validation_summary[column] = invalid_count
            else:
                # Generic validation for other columns
                def is_valid_general(x):
                    if pd.isna(x) or str(x).strip() == '':
                        return False
                    return True
                
                validated_df[f"{column}_Valid"] = df[column].apply(is_valid_general)
                invalid_count = (~validated_df[f"{column}_Valid"]).sum()
                validation_summary[column] = invalid_count
        
        # Display validation summary
        if validation_summary:
            st.subheader("📈 Validation Summary")
            cols = st.columns(len(validation_summary))
            
            for i, (column, invalid_count) in enumerate(validation_summary.items()):
                with cols[i]:
                    total_rows = len(df)
                    percentage = (invalid_count / total_rows) * 100
                    st.metric(
                        f"{column}",
                        f"{invalid_count} invalid",
                        f"{percentage:.1f}% of data"
                    )
        
        # Create editable grid with validation highlighting
        st.subheader("📝 Editable Data Table")
        st.write("Invalid cells are highlighted in red. Click on any cell to edit it.")
        
        # JavaScript code for cell styling based on validation
        cell_style_jscode = JsCode("""
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
                return {
                    'backgroundColor': '#c8e6c9',
                    'color': '#2e7d32'
                };
            }
            return {};
        }
        """)
        
        # Configure grid options
        gb = GridOptionsBuilder.from_dataframe(validated_df)
        
        # Make only the original columns editable (not the validation columns)
        original_columns = [col for col in df.columns]
        for col in validated_df.columns:
            if col.endswith('_Valid'):
                gb.configure_column(col, hide=True)  # Hide validation columns
            elif col in original_columns:
                gb.configure_column(col, editable=True, cellStyle=cell_style_jscode)
        
        gb.configure_grid_options(
            enableRangeSelection=True,
            enableCellTextSelection=True,
            ensureDomOrder=True
        )
        
        grid_options = gb.build()
        
        # Display the grid
        grid_response = AgGrid(
            validated_df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            allow_unsafe_jscode=True,
            fit_columns_on_grid_load=True,
            height=400,
            reload_data=False
        )
        
        # Get edited data
        if grid_response['data'] is not None:
            edited_df = pd.DataFrame(grid_response['data'])
            
            # Re-validate after edits
            for column in original_columns:
                if column == 'PhoneNumber':
                    edited_df[f"{column}_Valid"] = edited_df[column].apply(is_phone_number_valid)
                elif column == 'BloodSugar':
                    edited_df[f"{column}_Valid"] = edited_df[column].apply(is_blood_sugar_valid)
                else:
                    def is_valid_general(x):
                        if pd.isna(x) or str(x).strip() == '':
                            return False
                        return True
                    edited_df[f"{column}_Valid"] = edited_df[column].apply(is_valid_general)
            
            st.session_state.validated_df = edited_df
        
        # Correction suggestions section
        st.header("💡 AI Correction Suggestions")
        
        if st.session_state.validated_df is not None:
            current_data = st.session_state.validated_df
            
            for column in original_columns:
                if f"{column}_Valid" in current_data.columns:
                    invalid_mask = current_data[f"{column}_Valid"] == False
                    invalid_count = invalid_mask.sum()
                    
                    if invalid_count > 0:
                        with st.expander(f"🔧 {column} - {invalid_count} issues found"):
                            invalid_data = current_data[invalid_mask]
                            
                            corrections = []
                            for idx, row in invalid_data.iterrows():
                                original_value = row[column]
                                
                                if column == 'PhoneNumber':
                                    suggested = st.session_state.corrector.suggest_phone_correction(original_value)
                                elif column == 'BloodSugar':
                                    suggested = st.session_state.corrector.suggest_blood_sugar_correction(original_value)
                                else:
                                    suggested = st.session_state.corrector.suggest_general_correction(original_value, column)
                                
                                corrections.append({
                                    'Row': idx,
                                    'Original': original_value,
                                    'Suggested': suggested,
                                    'Apply': False
                                })
                            
                            if corrections:
                                corrections_df = pd.DataFrame(corrections)
                                st.dataframe(corrections_df, use_container_width=True)
                                
                                if st.button(f"Apply All {column} Corrections", key=f"apply_{column}"):
                                    applied_count = 0
                                    for correction in corrections:
                                        if correction['Suggested'] is not None:
                                            st.session_state.validated_df.loc[correction['Row'], column] = correction['Suggested']
                                            applied_count += 1
                                    
                                    if applied_count > 0:
                                        st.success(f"Applied {applied_count} corrections to {column}")
                                        st.rerun()
        
        # Export section
        st.header("📥 Export Cleaned Data")
        
        if st.session_state.validated_df is not None:
            # Get final data without validation columns
            export_df = st.session_state.validated_df[[col for col in original_columns]].copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    "📄 Download as CSV",
                    csv_data,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel export
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    export_df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
                
                st.download_button(
                    "📊 Download as Excel",
                    output.getvalue(),
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            # Show final statistics
            st.subheader("📊 Final Data Quality")
            total_cells = len(export_df) * len(export_df.columns)
            
            if st.session_state.validated_df is not None:
                valid_cells = 0
                for col in original_columns:
                    if f"{col}_Valid" in st.session_state.validated_df.columns:
                        valid_cells += st.session_state.validated_df[f"{col}_Valid"].sum()
                
                quality_percentage = (valid_cells / total_cells) * 100
                st.metric("Data Quality", f"{quality_percentage:.1f}%", f"{valid_cells}/{total_cells} valid cells")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please check your file format and try again.")

# Sample data section
st.header("📋 Load Sample Data")

sample_files = {
    "Phone Validation Sample": "data/sample_data/synthetic_testing_validation_phone.csv",
    "Health Data Sample": "data/sample_data/synthetic_testing_validation_health.csv",
    "Training Data Sample": "data/sample_data/synthetic_data_for_training_correction.csv"
}

cols = st.columns(len(sample_files))

for i, (name, file_path) in enumerate(sample_files.items()):
    with cols[i]:
        if st.button(name, use_container_width=True):
            if os.path.exists(file_path):
                sample_df = pd.read_csv(file_path)
                st.session_state.current_df = sample_df
                st.success(f"Loaded {name}")
                st.rerun()
            else:
                st.error(f"File not found: {file_path}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 ML Data Validator - Powered by Machine Learning & Rule-Based Validation</p>
    <p>Upload your data, get intelligent validation, and download cleaned results</p>
</div>
""", unsafe_allow_html=True)