import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import re
import joblib
import os


class PhoneValidatorML:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=1000)
        self.is_trained = False

    # ============================================================================
    # VALIDATION SECTION - Phone number validation and feature extraction
    # ============================================================================
        
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

    def predict(self, phone_numbers):
        """Predict if phone numbers are valid - MAIN VALIDATION METHOD"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(phone_numbers, str):
            phone_numbers = [phone_numbers]
        
        X = self.extract_features(phone_numbers)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities

    # ============================================================================
    # TRAINING SECTION - ML model training and data generation
    # ============================================================================
    
    def create_training_data(self, size=1000):
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
    
    def train(self, training_data=None):
        """Train the phone validation model"""
        if training_data is None:
            print("Creating synthetic training data...")
            df = self.create_training_data(2000)
        else:
            df = training_data.copy()
            
        # Extract features
        print("Extracting features...")
        X = self.extract_features(df['phone'])
        y = df['is_valid']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Training completed!")
        print(f"Model accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

    # ============================================================================
    # MODEL PERSISTENCE SECTION - Save and load trained models
    # ============================================================================
    
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
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if not os.path.exists(filepath):
            return False
        
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.is_trained = model_data['is_trained']
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# ============================================================================
# TRAINING UTILITIES - High-level training functions
# ============================================================================

def train_phone_model():
    """Train and save a phone validation model"""
    validator = PhoneValidatorML()
    
    # Train the model
    results = validator.train()
    
    # Save the model
    model_path = 'ml/trained_models/phone_validator_model.pkl'
    validator.save_model(model_path)
    
    return validator, results


if __name__ == "__main__":
    # Train and save the model
    print("Training Phone Validation ML Model...")
    model, results = train_phone_model()
    
    # Test with some examples
    test_phones = [
        "+1234567890",     # Valid
        "+65 9123 4567",   # Valid with spaces
        "invalid_phone",   # Invalid
        "123",             # Too short
        "+1234567890123456", # Too long
    ]
    
    predictions, probabilities = model.predict(test_phones)
    
    print("\nTest Results:")
    for phone, pred, prob in zip(test_phones, predictions, probabilities):
        validity = "Valid" if pred == 1 else "Invalid"
        confidence = max(prob) * 100
        print(f"{phone:<20} -> {validity} (confidence: {confidence:.1f}%)")