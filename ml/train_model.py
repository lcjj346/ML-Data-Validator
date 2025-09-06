"""
MAIN TRAINING INTERFACE - High-level functions to train phone validation models

QUICK START:
---------------------------------------------------------
To train a model with synthetic data (recommended):
    cd ml
    python train_model.py

TRAINING OPTIONS:

Option 1: Train with synthetic data (fastest, most reliable)
---------------------------------------------------------
from ml.train_model import train_phone_validation_model
trainer, results = train_phone_validation_model()
print(f"Accuracy: {results['accuracy']:.3f}")

Option 2: Train with your own CSV data
---------------------------------------------------------
# Your CSV must have columns: 'PhoneNumber' and 'PhoneNumber_Valid'
# PhoneNumber_Valid should be 1 for valid, 0 for invalid
from ml.train_model import train_from_csv_data
trainer, results = train_from_csv_data('path/to/your/data.csv')
if results:
    print(f"Accuracy: {results['accuracy']:.3f}")

VIEWING RESULTS:
---------------------------------------------------------
# After training, results contains:
results = {
    'accuracy': 0.999,  # Overall accuracy (0-1 scale)
    'classification_report': {
        '0': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0},  # Invalid numbers
        '1': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0},  # Valid numbers
        'accuracy': 0.999,
        'macro avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0}
    }
}

# View feature importance:
importance = trainer.get_feature_importance()
print("Most important features:")
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {score:.3f}")

MODEL OUTPUT:
---------------------------------------------------------
Trained model is automatically saved to: ml/trained_models/phone_validator_model.pkl
Use it with: from ml.phone_validator import PhoneValidator
"""

from phone_model_trainer import train_phone_model, train_from_csv_file
import pandas as pd
import os


def train_phone_validation_model():
    """Train the phone validation ML model using synthetic data"""
    print("Training phone validation model with synthetic data...")
    trainer, results = train_phone_model()
    return trainer, results


def train_from_csv_data(csv_file_path):
    """Train phone validation model from CSV data"""
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found: {csv_file_path}")
        return None
    
    try:
        trainer, results = train_from_csv_file(csv_file_path)
        return trainer, results
    except Exception as e:
        print(f"Error training from CSV: {e}")
        return None


if __name__ == "__main__":
    """
    MAIN ENTRY POINT - This runs when you execute: python train_model.py
    
    This will:
    1. Train a model using 2000 synthetic phone number examples
    2. Display accuracy and detailed metrics
    3. Show feature importance 
    4. Save the model for use with phone_validator.py
    
    Expected accuracy: Close to 100% (1.000)
    """
    print(">>> Starting phone validation model training...")
    print(">>> Using synthetic data generation method")
    print("-" * 50)
    
    # Train the default model
    trainer, results = train_phone_validation_model()
    
    # Display comprehensive results
    print(f"\n>>> FINAL RESULTS:")
    print(f"   Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    
    # Show what features matter most
    print(f"\n>>> TOP 5 MOST IMPORTANT FEATURES:")
    importance = trainer.get_feature_importance()
    for i, (feature, score) in enumerate(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5], 1):
        print(f"   {i}. {feature}: {score:.3f} ({score*100:.1f}%)")
    
    print(f"\n>>> Model training completed successfully!")
    print(f">>> Ready to use with: from ml.phone_validator import PhoneValidator")