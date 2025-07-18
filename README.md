# 🧠 ML-Data-Validator

This project is a Streamlit-based API that allows users to upload survey data (CSV/Excel), highlights invalid fields using AI, and suggests corrections. It is built to help clean up data before it's fed into analytics or reporting pipelines.

## 🔧 Features

- Upload `.csv` or `.xlsx` survey files
- Automatically detect formatting or logical issues (e.g., mistyped phone numbers)
- View and edit data in a user-friendly dashboard
- Machine Learning suggestions for corrections (e.g., fixing "+65 1234abc" to "+65 12345678")
- Download cleaned data after review
- Export `.csv` or `.xlsx` survey files

## 🚀 Technologies Used

- Streamlit
- Python
- Pandas / Openpyxl
- Scikit-learn (for ML validation)

## 📁 Files and Directory

ML-Data-Validator/
│

├── data/
│ └── synthetic_validation_dataset.csv # Sample dataset with common input mistakes
│
├── ml/
│ ├── train_model.py # Script to train the ML model using sample data
│ ├── model.pkl # Serialized trained model (generated after training)
│ └── validation_engine.py # Core logic to validate and suggest corrections using the model
│
├── app/
│ ├── streamlit_app.py # Main Streamlit web interface
│
├── assets/ # Images, screenshots
│ ├── UIDraft.jpg  
│ └── WorkflowDiagram.jpg  
│
├── predict.py # Generates predictions from the trained model (used by validation engine)
├── preprocess.py # Data cleaning and preprocessing functions for model input
├── requirements.txt # Python package dependencies
├── README.md # Project overview and documentation
└── .gitignore # Git configuration to ignore unnecessary files

## 🏁 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
