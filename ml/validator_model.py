import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from .data_preprocessor import DataPreprocessor

class MLValidator:
    """Machine Learning model for data validation and correction."""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.feature_columns = {}
        self.correction_models = {}
        
    def train_validation_model(self, df: pd.DataFrame, column: str, 
                             valid_column: str, feature_type: str = 'general') -> Dict[str, Any]:
        """Train a validation model for a specific column."""
        
        # Extract features
        features_df = self.preprocessor.extract_features(df, column, feature_type)
        
        # Prepare training data
        X = features_df
        y = df[valid_column].astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store model and metadata
        self.models[column] = model
        self.feature_columns[column] = list(features_df.columns)
        
        return {
            'accuracy': accuracy,
            'feature_importance': dict(zip(features_df.columns, model.feature_importances_)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def train_correction_model(self, df: pd.DataFrame, column: str, 
                             corrected_column: str, feature_type: str = 'general') -> Dict[str, Any]:
        """Train a correction model to suggest fixes for invalid data."""
        
        # Only train on valid data for correction
        valid_data = df[df[f"{column}_Valid"] == 1].copy()
        
        if len(valid_data) < 10:
            return {'error': 'Not enough valid data for training correction model'}
        
        # Extract features from original values
        features_df = self.preprocessor.extract_features(valid_data, column, feature_type)
        
        # Target is the corrected/clean version
        if feature_type == 'blood_sugar':
            # For blood sugar, predict the actual numeric value
            y = pd.to_numeric(valid_data[corrected_column], errors='coerce')
            model = LinearRegression()
        else:
            # For other types, we'll use a simple approach
            return {'error': 'Correction model not implemented for this type'}
        
        # Remove any NaN targets
        mask = ~pd.isna(y)
        X = features_df[mask]
        y = y[mask]
        
        if len(X) < 5:
            return {'error': 'Not enough clean data for correction model'}
        
        # Train correction model
        model.fit(X, y)
        
        # Store correction model
        self.correction_models[column] = model
        
        return {'success': True, 'training_samples': len(X)}
    
    def predict_validity(self, df: pd.DataFrame, column: str, feature_type: str = 'general') -> np.ndarray:
        """Predict validity for a column."""
        if column not in self.models:
            raise ValueError(f"No trained model found for column: {column}")
        
        # Extract features
        features_df = self.preprocessor.extract_features(df, column, feature_type)
        
        # Ensure feature columns match training
        expected_columns = self.feature_columns[column]
        features_df = features_df.reindex(columns=expected_columns, fill_value=0)
        
        # Predict
        predictions = self.models[column].predict(features_df)
        return predictions
    
    def suggest_corrections(self, df: pd.DataFrame, column: str, feature_type: str = 'general') -> np.ndarray:
        """Suggest corrections for invalid data."""
        if column not in self.correction_models:
            return np.array([None] * len(df))
        
        # Extract features
        features_df = self.preprocessor.extract_features(df, column, feature_type)
        
        # Predict corrections
        corrections = self.correction_models[column].predict(features_df)
        return corrections
    
    def save_models(self, filepath: str):
        """Save trained models to disk."""
        model_data = {
            'models': self.models,
            'feature_columns': self.feature_columns,
            'correction_models': self.correction_models
        }
        joblib.dump(model_data, filepath)
    
    def load_models(self, filepath: str):
        """Load trained models from disk."""
        # if os.path.exists(filepath):
        #     model_data = joblib.load(filepath)
        #     self.models = model_data.get('models', {})
        #     self.feature_columns = model_data.get('feature_columns', {})
        #     self.correction_models = model_data.get('correction_models', {})
        #     return True
        return False