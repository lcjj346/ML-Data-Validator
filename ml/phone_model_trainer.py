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
from sklearn.ensemble import RandomForestClassifier
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
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
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
    
    def create_synthetic_training_data(self, size=2000):
        """Create synthetic training data for phone validation"""
        np.random.seed(42)
        
        # Valid phone patterns
        valid_patterns = [
            '+1{}{}{}{}{}{}{}{}{}{}',   # US format - 10 digits
            '+65{}{}{}{}{}{}{}{}',      # Singapore format - 8 digits
            '+44{}{}{}{}{}{}{}{}{}',    # UK format - 9 digits
            '{}{}{}{}{}{}{}{}{}{}',     # 10 digit format
            '({}{}{}) {}{}{}-{}{}{}{}', # US with formatting - 10 digits
        ]
        
        # Invalid patterns
        invalid_patterns = [
            '{}{}{}abc',              # Too short with letters
            '+{}{}{}{}{}{}{}{}{}{}{}{}{}{}', # Too long
            '++{}{}{}{}{}{}{}{}',      # Double plus
            '',                       # Empty
            'not_a_phone',           # Text
            '123',                   # Too short
            '+1234567890123456789',  # Way too long
        ]
        
        data = []
        
        # Generate valid phone numbers
        for _ in range(size // 2):
            pattern = np.random.choice(valid_patterns)
            # Generate appropriate number of digits for each pattern
            if '+65' in pattern:
                digits = [str(np.random.randint(0, 10)) for _ in range(8)]
            elif '+44' in pattern:
                digits = [str(np.random.randint(0, 10)) for _ in range(9)]
            else:
                digits = [str(np.random.randint(0, 10)) for _ in range(10)]
            
            try:
                phone = pattern.format(*digits)
                data.append({'phone': phone, 'is_valid': 1})
            except IndexError:
                # Fallback to simple 10-digit format
                digits = [str(np.random.randint(0, 10)) for _ in range(10)]
                phone = ''.join(digits)
                data.append({'phone': phone, 'is_valid': 1})
        
        # Generate invalid phone numbers
        for _ in range(size // 2):
            if np.random.random() < 0.3:
                # Use invalid patterns
                pattern = np.random.choice(invalid_patterns)
                if '{}' in pattern:
                    digits = [str(np.random.randint(0, 10)) for _ in range(3)]
                    try:
                        phone = pattern.format(*digits)
                    except IndexError:
                        phone = pattern
                else:
                    phone = pattern
            else:
                # Corrupt valid patterns
                pattern = np.random.choice(valid_patterns)
                # Generate appropriate number of digits
                if '+65' in pattern:
                    digits = [str(np.random.randint(0, 10)) for _ in range(8)]
                elif '+44' in pattern:
                    digits = [str(np.random.randint(0, 10)) for _ in range(9)]
                else:
                    digits = [str(np.random.randint(0, 10)) for _ in range(10)]
                
                try:
                    phone = pattern.format(*digits)
                    # Add corruption
                    corruption_type = np.random.choice(['add_letters', 'remove_digits', 'add_symbols'])
                    if corruption_type == 'add_letters':
                        phone = phone + 'abc'
                    elif corruption_type == 'remove_digits':
                        phone = phone[:len(phone)//2]
                    else:  # add_symbols
                        phone = phone + '@#$'
                except IndexError:
                    phone = 'invalid_phone_' + str(np.random.randint(1000, 9999))
            
            data.append({'phone': phone, 'is_valid': 0})
        
        return pd.DataFrame(data)
    
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
        print("Training Random Forest model...")
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
    
    def train_from_synthetic_data(self, size=2000):
        """Train model using synthetic data generation"""
        print("Creating synthetic training data...")
        training_data = self.create_synthetic_training_data(size)
        return self.train_from_data(training_data)
    
    def train_from_csv(self, csv_file_path):
        """
        Train model from CSV file.
        CSV must have columns: 'PhoneNumber' and 'PhoneNumber_Valid'
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        print(f"Loading training data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        
        # Check required columns
        if 'PhoneNumber' not in df.columns or 'PhoneNumber_Valid' not in df.columns:
            raise ValueError("CSV must have 'PhoneNumber' and 'PhoneNumber_Valid' columns")
        
        # Prepare training data
        training_data = pd.DataFrame({
            'phone': df['PhoneNumber'],
            'is_valid': df['PhoneNumber_Valid']
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
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        feature_names = [
            'length', 'starts_with_plus', 'digit_count', 'non_digit_count',
            'has_spaces', 'has_dashes', 'has_parentheses', 'has_letters',
            'consecutive_digits', 'valid_length'
        ]
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        
        return feature_importance


# High-level training functions
def train_phone_model(save_path='ml/trained_models/phone_validator_model.pkl'):
    """
    Train and save a phone validation model using synthetic data.
    Returns: (trainer_instance, training_results)
    """
    print("Initializing Phone Model Trainer...")
    trainer = PhoneModelTrainer()
    
    # Train the model with synthetic data
    results = trainer.train_from_synthetic_data(2000)
    
    # Save the trained model
    trainer.save_model(save_path)
    
    print(f"\nPhone validation model training completed!")
    print(f"Final accuracy: {results['accuracy']:.3f}")
    
    return trainer, results


def train_from_csv_file(csv_path, save_path='ml/trained_models/phone_validator_model.pkl'):
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
    
    This runs when you execute: python phone_model_trainer.py
    
    It will:
    1. Create 2000 synthetic phone examples (1000 valid, 1000 invalid)
    2. Train a Random Forest model on these examples
    3. Show you the accuracy and detailed metrics
    4. Save the model to ml/trained_models/phone_validator_model.pkl
    5. Display which features are most important for classification
    
    Expected output:
    - Accuracy should be close to 1.000 (100%)
    - Most important features are usually: length, valid_length, digit_count
    """
    
    print("=" * 60)
    print("TRAINING PHONE VALIDATION ML MODEL")
    print("=" * 60)
    
    # Train the model and test it
    trainer, results = train_phone_model()
    
    # Show detailed results
    print(f"\n>>> FINAL ACCURACY: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    
    # Show feature importance (what the model learned is most important)
    print("\n>>> FEATURE IMPORTANCE (what the model considers most important):")
    print("-" * 50)
    importance = trainer.get_feature_importance()
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature:<20}: {score:.3f} ({score*100:.1f}%)")
    
    print(f"\n>>> Model saved and ready for use!")
    print(f">>> Location: ml/trained_models/phone_validator_model.pkl")
    print(f">>> Use it with: from phone_validator import PhoneValidator")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)