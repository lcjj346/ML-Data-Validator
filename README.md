# 🧠 AI Validation Chatbot for Survey Data

This project is a Streamlit-based chatbot that allows users to upload survey data (CSV/Excel), highlights invalid fields using AI, and suggests corrections. It is built to help clean up data before it's fed into analytics or reporting pipelines.

## 🔧 Features
- Upload `.csv` or `.xlsx` survey files
- Automatically detect formatting or logical issues (e.g., mistyped phone numbers)
- View and edit data in a user-friendly dashboard
- AI-powered suggestions for corrections (e.g., fixing "+65 1234abc" to "+65 12345678")
- Download cleaned data after review

## 🚀 Technologies Used
- Streamlit
- Python
- Pandas / Openpyxl
- Scikit-learn (for ML validation)
- Optional: OpenAI GPT (for natural-language explanation)

## 🏁 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
