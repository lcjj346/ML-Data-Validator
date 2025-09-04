import streamlit as st
import pandas as pd
import numpy as np
import os
from ml.validator_model import MLValidator
from ml.model_trainer import ModelTrainer
from ml.data_corrector import DataCorrector
from utils.phone_validator import is_phone_number_valid
from utils.blood_sugar_validator import is_blood_sugar_valid

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
    .invalid-cell {
        background-color: #ffebee !important;
        color: #c62828 !important;
    }
    .valid-cell {
        background-color: #e8f5e8 !important;
        color: #2e7d32 !important;
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
col1, col2 = st.columns([2, 1])

with col1:
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
            
            # Show original data preview
            with st.expander("📊 Original Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Data validation section
            st.header("🔍 AI-Powered Validation")
            
            if st.session_state.models_trained:
                # Create a copy for validation
                validated_df = df.copy()
                
                # Apply ML validation for known columns
                validation_results = {}
                
                for column in df.columns:
                    if column in ['PhoneNumber', 'BloodSugar']:
                        feature_type = 'phone' if column == 'PhoneNumber' else 'blood_sugar'
                        
                        try:
                            # Get ML predictions
                            ml_predictions = st.session_state.validator.predict_validity(
                                df, column, feature_type
                            )
                            validated_df[f"{column}_ML_Valid"] = ml_predictions
                            
                            # Get rule-based validation for comparison
                            if column == 'PhoneNumber':
                                rule_predictions = df[column].apply(is_phone_number_valid)
                            else:
                                rule_predictions = df[column].apply(is_blood_sugar_valid)
                            
                            validated_df[f"{column}_Rule_Valid"] = rule_predictions
                            
                            # Count invalid entries
                            ml_invalid = (ml_predictions == 0).sum()
                            rule_invalid = (~rule_predictions).sum()
                            
                            validation_results[column] = {
                                'ml_invalid': ml_invalid,
                                'rule_invalid': rule_invalid,
                                'total_rows': len(df)
                            }
                            
                        except Exception as e:
                            st.warning(f"Could not validate {column} with ML: {str(e)}")
                
                # Display validation results
                if validation_results:
                    st.subheader("📈 Validation Results")
                    
                    for column, results in validation_results.items():
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric(
                                f"{column} - ML Invalid",
                                results['ml_invalid'],
                                f"{results['ml_invalid']/results['total_rows']*100:.1f}%"
                            )
                        
                        with col_b:
                            st.metric(
                                f"{column} - Rule Invalid", 
                                results['rule_invalid'],
                                f"{results['rule_invalid']/results['total_rows']*100:.1f}%"
                            )
                        
                        with col_c:
                            agreement = np.mean(
                                validated_df[f"{column}_ML_Valid"] == validated_df[f"{column}_Rule_Valid"].astype(int)
                            )
                            st.metric("ML-Rule Agreement", f"{agreement:.3f}")
                
                # Show validated data with highlighting
                st.subheader("🎯 Validated Data with Highlights")
                
                # Create a styled dataframe
                def highlight_invalid(row):
                    styles = [''] * len(row)
                    for i, (col, value) in enumerate(row.items()):
                        if col.endswith('_ML_Valid'):
                            continue
                        
                        # Check if there's a corresponding validity column
                        valid_col = f"{col}_ML_Valid"
                        if valid_col in validated_df.columns:
                            is_valid = row.get(valid_col, True)
                            if not is_valid:
                                styles[i] = 'background-color: #ffcdd2; color: #d32f2f'
                    return styles
                
                # Display only original columns for clarity
                display_df = validated_df[[col for col in df.columns]]
                
                # Apply styling
                styled_df = display_df.style.apply(highlight_invalid, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                
                # Correction suggestions
                st.subheader("💡 AI Correction Suggestions")
                
                corrections_made = False
                corrected_df = df.copy()
                
                for column in df.columns:
                    if column in validation_results:
                        invalid_mask = validated_df[f"{column}_ML_Valid"] == 0
                        invalid_count = invalid_mask.sum()
                        
                        if invalid_count > 0:
                            st.write(f"**{column}**: {invalid_count} invalid entries found")
                            
                            # Generate corrections
                            corrections = []
                            for idx, (_, row) in enumerate(df[invalid_mask].iterrows()):
                                original_value = row[column]
                                
                                if column == 'PhoneNumber':
                                    suggested = st.session_state.corrector.suggest_phone_correction(original_value)
                                elif column == 'BloodSugar':
                                    suggested = st.session_state.corrector.suggest_blood_sugar_correction(original_value)
                                else:
                                    suggested = st.session_state.corrector.suggest_general_correction(original_value, column)
                                
                                corrections.append({
                                    'Row': row.name,
                                    'Original': original_value,
                                    'Suggested': suggested
                                })
                            
                            if corrections:
                                corrections_df = pd.DataFrame(corrections)
                                st.dataframe(corrections_df, use_container_width=True)
                                
                                # Apply corrections button
                                if st.button(f"Apply All {column} Corrections"):
                                    for correction in corrections:
                                        if correction['Suggested'] is not None:
                                            corrected_df.loc[correction['Row'], column] = correction['Suggested']
                                    corrections_made = True
                                    st.success(f"Applied {len(corrections)} corrections to {column}")
                
                # Export section
                if corrections_made or st.checkbox("Export current data"):
                    st.subheader("📥 Export Cleaned Data")
                    
                    export_format = st.selectbox("Choose export format", ["CSV", "Excel"])
                    
                    if export_format == "CSV":
                        csv_data = corrected_df.to_csv(index=False)
                        st.download_button(
                            "Download Cleaned CSV",
                            csv_data,
                            file_name="cleaned_data.csv",
                            mime="text/csv"
                        )
                    else:
                        # Convert to Excel
                        import io
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            corrected_df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
                        
                        st.download_button(
                            "Download Cleaned Excel",
                            output.getvalue(),
                            file_name="cleaned_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            else:
                st.warning("⚠️ Please train or load ML models first using the sidebar options")
                
                # Fallback to rule-based validation
                st.subheader("📋 Rule-Based Validation (Fallback)")
                
                rule_validated_df = df.copy()
                
                # Apply rule-based validation
                if 'PhoneNumber' in df.columns:
                    rule_validated_df['PhoneNumber_Valid'] = df['PhoneNumber'].apply(is_phone_number_valid)
                    invalid_phones = (~rule_validated_df['PhoneNumber_Valid']).sum()
                    st.metric("Invalid Phone Numbers", invalid_phones)
                
                if 'BloodSugar' in df.columns:
                    rule_validated_df['BloodSugar_Valid'] = df['BloodSugar'].apply(is_blood_sugar_valid)
                    invalid_blood_sugar = (~rule_validated_df['BloodSugar_Valid']).sum()
                    st.metric("Invalid Blood Sugar Values", invalid_blood_sugar)
                
                # Show rule-based results
                display_columns = [col for col in rule_validated_df.columns if not col.endswith('_Valid')]
                st.dataframe(rule_validated_df[display_columns], use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with col2:
    st.header("📊 Model Statistics")
    
    if st.session_state.models_trained:
        st.success("✅ Models are trained and ready")
        
        # Show model info
        if hasattr(st.session_state.validator, 'models'):
            for model_name in st.session_state.validator.models.keys():
                with st.expander(f"📈 {model_name} Model"):
                    st.write("**Model Type**: Random Forest Classifier")
                    st.write("**Features**: Extracted automatically from data patterns")
                    st.write("**Status**: Ready for prediction")
    else:
        st.info("🔄 No models loaded. Use sidebar to train or load models.")
    
    # Sample data info
    st.header("📋 Sample Data")
    
    sample_files = [
        "data/sample_data/synthetic_data_for_training_correction.csv",
        "data/sample_data/synthetic_testing_validation_phone.csv",
        "data/sample_data/synthetic_testing_validation_health.csv"
    ]
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            if st.button(f"Load {file_name}"):
                sample_df = pd.read_csv(file_path)
                st.session_state.sample_data = sample_df
                st.success(f"Loaded {file_name}")
                st.dataframe(sample_df.head(), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 ML Data Validator - Powered by Machine Learning & Rule-Based Validation</p>
    <p>Upload your data, get intelligent validation, and download cleaned results</p>
</div>
""", unsafe_allow_html=True)