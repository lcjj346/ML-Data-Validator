"""
PHONE MODEL TRAINER - Train ML models for phone number validation

HOW TO TRAIN A MODEL:

Method 1: Train with synthetic data (recommended for general use)
---------------------------------------------------------
cd ml
python phone_model_trainer.py

Method 2: Train from your own CSV data
---------------------------------------------------------
# Your CSV must have columns: 'PhoneNumber' and 'PhoneNumber_Valid'
# Example CSV format:
# PhoneNumber,PhoneNumber_Valid
# +1234567890,1
# invalid_phone,0
# +6591234567,1

from phone_model_trainer import train_from_csv_file
trainer, results = train_from_csv_file('path/to/your/data.csv')
print(f"Accuracy: {results['accuracy']:.3f}")

Method 3: Using the high-level training functions
---------------------------------------------------------
from phone_model_trainer import train_phone_model
trainer, results = train_phone_model()

HOW TO VIEW ACCURACY AND RESULTS:
---------------------------------------------------------
# After training, results contains:
# - results['accuracy']: Overall accuracy (0.0 to 1.0)
# - results['classification_report']: Detailed metrics per class

# To see feature importance (what the model considers most important):
importance = trainer.get_feature_importance()
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.3f}")

WHERE MODELS ARE SAVED:
---------------------------------------------------------
Default location: ml/trained_models/phone_validator_model.pkl
The trained model is automatically saved and can be used by phone_validator.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import joblib
import os


class PhoneModelTrainer:
    """
    Phone validation model trainer.
    This class is focused purely on training ML models for phone validation.
    """
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.is_trained = False
        
    def extract_features(self, phone_numbers):
        """Extract features from phone numbers for ML training"""
        features = []
        
        for phone in phone_numbers:
            phone_str = str(phone) if phone is not None else ""
            
            # Basic features
            feature_dict = {
                'length': len(phone_str),
                'starts_with_plus': int(phone_str.startswith('+')),
                'digit_count': len(re.findall(r'\d', phone_str)),
                'non_digit_count': len(re.findall(r'\D', phone_str)),
                'has_spaces': int(' ' in phone_str),
                'has_dashes': int('-' in phone_str),
                'has_parentheses': int('(' in phone_str or ')' in phone_str),
                'has_letters': int(bool(re.search(r'[a-zA-Z]', phone_str))),
                'consecutive_digits': self._count_consecutive_digits(phone_str),
                'valid_length': int(8 <= len(re.findall(r'\d', phone_str)) <= 15),
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _count_consecutive_digits(self, phone_str):
        """Count maximum consecutive digits"""
        matches = re.findall(r'\d+', phone_str)
        if matches:
            return max(len(match) for match in matches)
        return 0
    
    def train_from_data(self, training_data):
        """
        Train the phone validation model from provided data.
        training_data should be a DataFrame with columns: 'phone', 'is_valid'
        """
        print("Extracting features for training...")
        X = self.extract_features(training_data['phone'])
        y = training_data['is_valid']
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        print("Training Logistic Regression model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed!")
        print(f"Model accuracy: {accuracy:.3f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def train_from_default_data(self):
        """Train model from saved training data file"""
        default_path = '../data/logistic_regression_training.csv'

        if not os.path.exists(default_path):
            raise FileNotFoundError(
                f"Training data not found at {default_path}\n"
                f"Please run 'python generate_training_data.py' first to create the training data."
            )

        print(f"Loading training data from {default_path}...")
        df = pd.read_csv(default_path)

        # Prepare training data
        training_data = pd.DataFrame({
            'phone': df['phone'],
            'is_valid': df['is_valid']
        })

        print(f"Loaded {len(training_data)} training examples")
        return self.train_from_data(training_data)
    
    def train_from_csv(self, csv_file_path):
        """
        Train model from CSV file.
        CSV must have columns: 'PhoneNumber' and 'Valid'
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        print(f"Loading training data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)

        # Check required columns
        if 'PhoneNumber' not in df.columns or 'Valid' not in df.columns:
            raise ValueError("CSV must have 'PhoneNumber' and 'Valid' columns")

        # Prepare training data
        training_data = pd.DataFrame({
            'phone': df['PhoneNumber'],
            'is_valid': df['Valid']
        })
        
        print(f"Training on {len(training_data)} examples from CSV...")
        return self.train_from_data(training_data)
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Trained model saved to {filepath}")
    
    def get_feature_coefficients(self):
        """Get feature coefficients from the trained logistic regression model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        feature_names = [
            'length', 'starts_with_plus', 'digit_count', 'non_digit_count',
            'has_spaces', 'has_dashes', 'has_parentheses', 'has_letters',
            'consecutive_digits', 'valid_length'
        ]

        coefficients = self.model.coef_[0]  # Get coefficients for logistic regression
        feature_coefficients = dict(zip(feature_names, coefficients))

        return feature_coefficients


# High-level training functions
def train_phone_model(save_path='../saved_models/phone_validator_model.pkl'):
    """
    Train and save a phone validation model from saved training data.
    Returns: (trainer_instance, training_results)
    """
    print("Initializing Phone Model Trainer...")
    trainer = PhoneModelTrainer()

    # Train the model from saved data
    results = trainer.train_from_default_data()

    # Save the trained model
    trainer.save_model(save_path)

    print(f"\nPhone validation model training completed!")
    print(f"Final accuracy: {results['accuracy']:.3f}")

    return trainer, results


def train_from_csv_file(csv_path, save_path='../saved_models/phone_validator_model.pkl'):
    """
    Train and save a phone validation model from CSV file.
    Returns: (trainer_instance, training_results)
    """
    print("Initializing Phone Model Trainer...")
    trainer = PhoneModelTrainer()
    
    # Train from CSV
    results = trainer.train_from_csv(csv_path)
    
    # Save the trained model
    trainer.save_model(save_path)
    
    print(f"\nPhone validation model training from CSV completed!")
    print(f"Final accuracy: {results['accuracy']:.3f}")
    
    return trainer, results


if __name__ == "__main__":
    """
    MAIN TRAINING SCRIPT

    This runs when you execute: python model_trainer.py

    It will:
    1. Load training data from data/logistic_regression_training.csv
    2. Train a Logistic Regression model on these examples
    3. Show you the accuracy and detailed metrics
    4. Save the model to saved_models/phone_validator_model.pkl
    5. Display which features are most important for classification

    Note: If training data doesn't exist, run 'python generate_training_data.py' first

    Expected output:
    - Accuracy should be close to 1.000 (100%)
    - Most important features (highest absolute coefficients) are usually: length, valid_length, digit_count
    """
    
    print("=" * 60)
    print("TRAINING PHONE VALIDATION ML MODEL")
    print("=" * 60)
    
    # Train the model and test it
    trainer, results = train_phone_model()
    
    # Show detailed results
    print(f"\n>>> FINAL ACCURACY: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    
    # Show feature importance (what the model learned is most important)
    print("\n>>> MODEL COEFFICIENTS (feature weights):")
    print("-" * 50)
    coefficients = trainer.get_feature_coefficients()
    for feature, coef in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{feature:<20}: {coef:.3f}")
    
    print(f"\n>>> Model saved and ready for use!")
    print(f">>> Location: ml/trained_models/phone_validator_model.pkl")
    print(f">>> Use it with: from phone_validator import PhoneValidator")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)