"""
Phone Model Trainer - Train ML models for phone number validation.

This module provides training functionality for Logistic Regression phone validators.
Features are extracted using the centralized PhoneFeatureExtractor class.

Example usage:
    from ml.model_trainer import PhoneModelTrainer

    trainer = PhoneModelTrainer()
    results = trainer.train_from_default_data()
    trainer.save_model('saved_models/phone_validator_model.pkl')
    print(f"Accuracy: {results['accuracy']:.1%}")
"""

import os
from typing import Dict, Tuple, Any
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ml.core.phone_features import PhoneFeatureExtractor


class PhoneModelTrainer:
    """
    Phone validation model trainer using Logistic Regression.

    This class focuses purely on training ML models for phone validation.
    Features are extracted using PhoneFeatureExtractor.

    Attributes:
        model: LogisticRegression instance
        is_trained: Boolean flag indicating if model has been trained
    """

    def __init__(self):
        """Initialize PhoneModelTrainer with a Logistic Regression model."""
        self.model: LogisticRegression = LogisticRegression(random_state=42, max_iter=1000)
        self.is_trained: bool = False
    
    def train_from_data(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the phone validation model from provided data.

        Args:
            training_data: DataFrame with columns 'phone' and 'is_valid'

        Returns:
            Dictionary containing:
                - accuracy: Overall accuracy score (0.0 to 1.0)
                - classification_report: Detailed metrics dict

        Raises:
            KeyError: If required columns are missing from training_data
        """
        print("Extracting features for training...")
        X = PhoneFeatureExtractor.extract_features(training_data['phone'].tolist())
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

    def train_from_default_data(self) -> Dict[str, Any]:
        """
        Train model from saved training data file.

        Returns:
            Dictionary containing training results (accuracy, classification report)

        Raises:
            FileNotFoundError: If default training data file doesn't exist
        """
        default_path = '../data/logistic_regression_training.csv'

        if not os.path.exists(default_path):
            raise FileNotFoundError(
                f"Training data not found at {default_path}\n"
                f"Please run 'python generate_aligned_training_data.py' first."
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

    def train_from_csv(self, csv_file_path: str) -> Dict[str, Any]:
        """
        Train model from CSV file.

        Args:
            csv_file_path: Path to CSV file with 'PhoneNumber' and 'Valid' columns

        Returns:
            Dictionary containing training results

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
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

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to file.

        Args:
            filepath: Path where model should be saved (.pkl file)

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, filepath)
        print(f"Trained model saved to {filepath}")
    
    def get_feature_coefficients(self) -> Dict[str, float]:
        """
        Get feature coefficients from the trained logistic regression model.

        Returns:
            Dictionary mapping feature names to their coefficients

        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        feature_names = [
            'length', 'starts_with_plus', 'digit_count', 'non_digit_count',
            'has_spaces', 'has_dashes', 'has_parentheses', 'has_letters',
            'consecutive_digits', 'valid_length'
        ]

        coefficients = self.model.coef_[0]
        feature_coefficients = dict(zip(feature_names, coefficients))

        return feature_coefficients


# High-level training functions for convenience
def train_phone_model(save_path: str = '../saved_models/phone_validator_model.pkl') -> Tuple[PhoneModelTrainer, Dict[str, Any]]:
    """
    Train and save a phone validation model from saved training data.

    Convenience function that handles the full training pipeline.

    Args:
        save_path: Path where trained model should be saved

    Returns:
        Tuple of (trainer_instance, training_results)
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


def train_from_csv_file(
    csv_path: str,
    save_path: str = '../saved_models/phone_validator_model.pkl'
) -> Tuple[PhoneModelTrainer, Dict[str, Any]]:
    """
    Train and save a phone validation model from CSV file.

    Convenience function that handles the full training pipeline from custom data.

    Args:
        csv_path: Path to CSV file with training data
        save_path: Path where trained model should be saved

    Returns:
        Tuple of (trainer_instance, training_results)
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