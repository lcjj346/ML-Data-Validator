# ML-Data-Validator

This project is a Streamlit-based API that allows users to upload survey data (CSV/Excel), highlights invalid fields using AI, and suggests corrections. It is built to help clean up data before it's fed into analytics or reporting pipelines.

## Features

- Upload `.csv` or `.xlsx` survey files
- Automatically detect formatting or logical issues (e.g., mistyped phone numbers)
- View and edit data in a user-friendly dashboard
- Machine Learning suggestions for corrections (e.g., fixing "+65 1234abc" to "+65 12345678")
- Download cleaned data after review
- Export `.csv` or `.xlsx` survey files

## Technologies Used

- Streamlit
- Python
- Pandas / Openpyxl
- Scikit-learn (for ML validation)

## Files and Directory

ML-Data-Validator/

assets/ # UI images & workflow diagrams

data/

- processed_data/
- sample_data/
  - synthetic_data_for_testing_validation.csv
  - synthetic_data_for_training_correction.csv

ml/

- correction_engine.py # Suggests corrected values for invalid data
- model.pk1 # Trained ML model (binary classification)
- train_model.py # Script to train the model
- validation_engine.py # Validation pipeline using rules + ML

utils/ # Utility scripts for validators & preprocessing

- blood_sugar_validator.py
- phone_validator.py
- preprocess.py

- requirements.txt # Python package dependencies
- README.md # Documentation
- .gitignore # Git ignore rules
- streamlit_v2.py # Alternate Streamlit version
- streamlit_app.py # Main Streamlit UI

## How to Run Locally

```bash
pip install -r requirements.txt
or
python -m pip install -r requirements.txt #if pip install doesn't work
streamlit run streamlit_app.py
```
