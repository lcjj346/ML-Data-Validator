import streamlit as st
import pandas as pd
from utils.phone_validator import is_phone_number_valid
from utils.blood_sugar_validator import is_blood_sugar_valid

st.set_page_config(page_title="ML-Data-Validator", layout="wide")
st.title("ML-Data-Validator")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Remove extra valid-flag columns if they exist
        df = df.drop(columns=[col for col in df.columns if "Valid" in col], errors="ignore")

        # Count invalid entries
        # Issue - if column not found, error
        invalid_phone_count = df["PhoneNumber"].apply(lambda x: not is_phone_number_valid(x)).sum()
        invalid_blood_sugar_count = df["BloodSugar"].apply(lambda x: not is_blood_sugar_valid(x)).sum()

        

        # Apply highlighting
        def highlight_invalid(val, col_name):
            if col_name == "PhoneNumber" and not is_phone_number_valid(val):
                return "border: 2px solid red; background-color: #2b0000;"
            elif col_name == "BloodSugar" and not is_blood_sugar_valid(val):
                return "border: 2px solid red; background-color: #2b0000;"
            return ""

        styled_df = df.style.apply(
            lambda row: [
                highlight_invalid(row[col], col) for col in df.columns
            ],
            axis=1
        )

        # Show table
        st.subheader("Validated Data")
        st.dataframe(styled_df)

        # Summary section
        st.subheader("Detected Issues")
        st.write(f"Invalid phone numbers: **{invalid_phone_count}**")
        st.write(f"Invalid blood sugar entries: **{invalid_blood_sugar_count}**")

    except Exception as e:
        st.error(f"Error reading file: {e}")
