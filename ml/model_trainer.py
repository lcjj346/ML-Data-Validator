import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Dict, Any
import joblib
import os
from .validator_model import MLValidator
from .data_corrector import DataCorrector

class ModelTrainer:
    """Handles training of ML models for data validation."""
    
    def __init__(self):
        self.validator = MLValidator()
        self.corrector = DataCorrector()
        
    def train_from_sample_data(self) -> Dict[str, Any]:
        """Train models using the sample data files."""
        results = {}
        
        # Load training data
        training_file = "data/sample_data/synthetic_data_for_training_correction.csv"
        if not os.path.exists(training_file):
            return {'error': 'Training data file not found'}
        
        df = pd.read_csv(training_file)
        
        # Train phone validation model
        if 'PhoneNumber' in df.columns and 'PhoneNumber_Valid' in df.columns:
            phone_results = self.validator.train_validation_model(
                df, 'PhoneNumber', 'PhoneNumber_Valid', 'phone'
            )
            results['phone_validation'] = phone_results
        
        # Train blood sugar validation model
        if 'BloodSugar' in df.columns and 'BloodSugar_Valid' in df.columns:
            blood_sugar_results = self.validator.train_validation_model(
                df, 'BloodSugar', 'BloodSugar_Valid', 'blood_sugar'
            )
            results['blood_sugar_validation'] = blood_sugar_results
            
            # Train blood sugar correction model
            correction_results = self.validator.train_correction_model(
                df, 'BloodSugar', 'BloodSugar', 'blood_sugar'
            )
            results['blood_sugar_correction'] = correction_results
        
        # Save models
        os.makedirs('ml/trained_models', exist_ok=True)
        self.validator.save_models('ml/trained_models/validator_models.pkl')
        
        return results
    
    def evaluate_model_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance on validation data."""
        results = {}
        
        # Test phone validation
        if 'PhoneNumber' in df.columns and 'PhoneNumber_Valid' in df.columns:
            try:
                predictions = self.validator.predict_validity(df, 'PhoneNumber', 'phone')
                actual = df['PhoneNumber_Valid'].astype(int)
                accuracy = accuracy_score(actual, predictions)
                results['phone_accuracy'] = accuracy
            except Exception as e:
                results['phone_error'] = str(e)
        
        # Test blood sugar validation
        if 'BloodSugar' in df.columns and 'BloodSugar_Valid' in df.columns:
            try:
                predictions = self.validator.predict_validity(df, 'BloodSugar', 'blood_sugar')
                actual = df['BloodSugar_Valid'].astype(int)
                accuracy = accuracy_score(actual, predictions)
                results['blood_sugar_accuracy'] = accuracy
            except Exception as e:
                results['blood_sugar_error'] = str(e)
        
        return results