"""
Model Training UI for Streamlit App
Allows users to upload training data and train models directly from the UI
"""

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ml.model_trainer import PhoneModelTrainer
from ml.edit_distance_corrector import EditDistanceCorrector


def train_logistic_regression_model(training_data_df, progress_callback=None):
    """
    Train Logistic Regression model from DataFrame

    Args:
        training_data_df: DataFrame with columns 'phone' and 'is_valid'
        progress_callback: Function to call with progress updates

    Returns:
        dict with training results
    """
    try:
        if progress_callback:
            progress_callback("Initializing trainer...")

        trainer = PhoneModelTrainer()

        # Prepare data
        training_data = pd.DataFrame({
            'phone': training_data_df['phone'],
            'is_valid': training_data_df['is_valid']
        })

        if progress_callback:
            progress_callback(f"Training on {len(training_data)} samples...")

        # Train
        results = trainer.train_from_data(training_data)

        if progress_callback:
            progress_callback("Saving model...")

        # Save model
        save_path = 'saved_models/phone_validator_model.pkl'
        trainer.save_model(save_path)

        if progress_callback:
            progress_callback("Training complete!")

        return {
            'success': True,
            'accuracy': results['accuracy'],
            'report': results['classification_report'],
            'model_path': save_path
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def train_xgboost_model(training_data_df, progress_callback=None):
    """
    Train XGBoost Edit Distance Corrector from DataFrame

    Args:
        training_data_df: DataFrame with columns 'invalid', 'valid', 'operations'
        progress_callback: Function to call with progress updates

    Returns:
        dict with training results
    """
    try:
        if progress_callback:
            progress_callback("Initializing XGBoost corrector...")

        corrector = EditDistanceCorrector()

        # Convert DataFrame to training_data format
        training_data = []
        for _, row in training_data_df.iterrows():
            training_data.append({
                'invalid': row['invalid'],
                'valid': row['valid'],
                'operations': row['operations'].split('|')  # Convert pipe-separated string to list
            })

        if progress_callback:
            progress_callback(f"Preparing features from {len(training_data)} samples...")

        # Prepare features
        X, y = corrector.prepare_training_data(training_data)

        if progress_callback:
            progress_callback("Training XGBoost model...")

        # Train model (inline version of train_from_csv without CSV loading)
        unique_labels = pd.unique(y)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y_remapped = pd.array([label_map[label] for label in y])

        corrector.inference_label_map = {new: old for old, new in label_map.items()}

        import xgboost as xgb
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

        if progress_callback:
            progress_callback("Saving model...")

        # Save model
        save_path = 'saved_models/edit_distance_corrector.pkl'
        corrector.save(save_path)

        if progress_callback:
            progress_callback("Training complete!")

        return {
            'success': True,
            'accuracy': accuracy,
            'num_samples': len(training_data),
            'model_path': save_path
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def render_training_ui():
    """Render the model training UI"""
    st.header("🎯 Model Training Center")

    st.markdown("""
    Train or retrain ML models with your own data. Upload training data files to improve model accuracy.
    """)

    # Create tabs for different models
    tab1, tab2, tab3 = st.tabs([
        "📊 Logistic Regression (Validator)",
        "🔧 XGBoost (Corrector)",
        "📖 Training Guide"
    ])

    # Tab 1: Logistic Regression Training
    with tab1:
        st.subheader("Phone Number Validator (Logistic Regression)")

        st.info("""
        **Required CSV Format:**
        - Column 1: `phone` - Phone number string
        - Column 2: `is_valid` - 1 for valid, 0 for invalid

        **Example:**
        ```
        phone,is_valid
        +1234567890,1
        invalid_phone,0
        +6591234567,1
        ```
        """)

        # File uploader
        lr_file = st.file_uploader(
            "Upload Training Data (CSV)",
            type=['csv'],
            key='lr_upload',
            help="CSV file with 'phone' and 'is_valid' columns"
        )

        if lr_file is not None:
            try:
                # Read and preview data
                df = pd.read_csv(lr_file)

                # Validate columns
                if 'phone' not in df.columns or 'is_valid' not in df.columns:
                    st.error("❌ CSV must have 'phone' and 'is_valid' columns")
                else:
                    st.success(f"✅ Loaded {len(df)} training samples")

                    # Show preview
                    with st.expander("📋 Preview Data"):
                        st.dataframe(df.head(20))

                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Samples", len(df))
                        with col2:
                            valid_count = df['is_valid'].sum()
                            st.metric("Valid Samples", valid_count)
                        with col3:
                            invalid_count = len(df) - valid_count
                            st.metric("Invalid Samples", invalid_count)

                    # Train button
                    if st.button("🚀 Train Logistic Regression Model", type="primary", key='train_lr'):
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()

                        def update_progress(message):
                            progress_placeholder.info(f"⏳ {message}")

                        with st.spinner("Training in progress..."):
                            results = train_logistic_regression_model(df, update_progress)

                        progress_placeholder.empty()

                        if results['success']:
                            st.success("✅ Training completed successfully!")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Accuracy", f"{results['accuracy']:.1%}")
                            with col2:
                                st.metric("Model Saved", "✓")

                            st.info(f"📁 Model saved to: `{results['model_path']}`")

                            with st.expander("📊 Detailed Classification Report"):
                                st.json(results['report'])

                            st.balloons()
                        else:
                            st.error(f"❌ Training failed: {results['error']}")

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    # Tab 2: XGBoost Training
    with tab2:
        st.subheader("Phone Number Corrector (XGBoost)")

        st.info("""
        **Required CSV Format:**
        - Column 1: `invalid` - Invalid phone number
        - Column 2: `valid` - Corrected phone number
        - Column 3: `operations` - Edit operations (pipe-separated: keep|delete|replace_0, etc.)

        **Example:**
        ```
        invalid,valid,operations
        1234567890,+11234567890,insert_country|keep|keep|keep|...
        +1234567e90,+1234567890,keep|keep|...|replace_8|keep
        ```
        """)

        # File uploader
        xgb_file = st.file_uploader(
            "Upload Training Data (CSV)",
            type=['csv'],
            key='xgb_upload',
            help="CSV file with 'invalid', 'valid', and 'operations' columns"
        )

        if xgb_file is not None:
            try:
                # Read and preview data
                df = pd.read_csv(xgb_file)

                # Validate columns
                required_cols = ['invalid', 'valid', 'operations']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"❌ CSV must have columns: {', '.join(required_cols)}")
                else:
                    st.success(f"✅ Loaded {len(df)} training samples")

                    # Show preview
                    with st.expander("📋 Preview Data"):
                        st.dataframe(df.head(20))

                        st.metric("Total Samples", len(df))

                    # Train button
                    if st.button("🚀 Train XGBoost Model", type="primary", key='train_xgb'):
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()

                        def update_progress(message):
                            progress_placeholder.info(f"⏳ {message}")

                        with st.spinner("Training in progress..."):
                            results = train_xgboost_model(df, update_progress)

                        progress_placeholder.empty()

                        if results['success']:
                            st.success("✅ Training completed successfully!")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Training Accuracy", f"{results['accuracy']:.1%}")
                            with col2:
                                st.metric("Model Saved", "✓")

                            st.info(f"📁 Model saved to: `{results['model_path']}`")

                            st.balloons()
                        else:
                            st.error(f"❌ Training failed: {results['error']}")

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    # Tab 3: Training Guide
    with tab3:
        st.subheader("📖 Training Guide")

        st.markdown("""
        ### How to Train Models

        #### Option 1: Use Existing Synthetic Data

        1. Generate training data files:
        ```bash
        cd ml
        python generate_training_data.py
        ```

        2. Upload the generated CSV files:
           - `data/logistic_regression_training.csv` → Logistic Regression tab
           - `data/xgboost_corrector_training.csv` → XGBoost tab

        #### Option 2: Use Your Own Data

        1. Prepare your CSV file in the correct format (see each tab for details)
        2. Upload the file in the corresponding tab
        3. Click the "Train" button
        4. Wait for training to complete
        5. The new model will automatically replace the old one

        ### Best Practices

        - **Data Quality**: Ensure your training data is accurate and representative
        - **Data Size**: Minimum 500 samples recommended, 2000+ preferred
        - **Balance**: Include roughly equal valid/invalid samples for Logistic Regression
        - **Diversity**: Include various phone formats and error types
        - **Validation**: After training, test the model with real data

        ### Model Files Location

        Trained models are saved to:
        - Logistic Regression: `saved_models/phone_validator_model.pkl`
        - XGBoost Corrector: `saved_models/edit_distance_corrector.pkl`

        ### Retraining Strategy

        - Retrain when you have new patterns to learn
        - Retrain when model accuracy drops
        - Keep backup of old models before retraining
        - Test new models before deploying to production

        ### Troubleshooting

        **"CSV format error"**: Check that column names match exactly

        **"Training failed"**: Check for missing values or invalid data types

        **"Low accuracy"**: Increase training data size or improve data quality

        **"Model not loading"**: Restart the Streamlit app after training
        """)

        st.divider()

        st.markdown("""
        ### Quick Links

        - 📂 [Training Data Generator Script](ml/generate_training_data.py)
        - 🔧 [Model Trainer](ml/model_trainer.py)
        - 🎯 [XGBoost Corrector](ml/edit_distance_corrector.py)
        """)


if __name__ == "__main__":
    st.set_page_config(page_title="Model Training", page_icon="🎯", layout="wide")
    render_training_ui()
