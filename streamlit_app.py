import streamlit as st
import pandas as pd
from utils.phone_validator import is_phone_number_valid  # Import our validation function

st.set_page_config(page_title="ML-Data-Validator", layout="wide")
st.title("ML-Data-Validator")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    try:
        # Load CSV or Excel
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Uploaded Data")

        # ---- Phone Number Validation ----
        if "PhoneNumber" in df.columns:
            # Apply our custom validation function
            df["PhoneNumber_Valid_Flag"] = df["PhoneNumber"].apply(is_phone_number_valid)

            # Function to highlight invalid phone numbers with red border
            def highlight_invalid_phone(val, valid_flag):
                return 'background-color: rgba(255, 0, 0, 0.2); color: white;' if not valid_flag else ''

            # Apply styling to dataframe
            styled_df = df.style.apply(
                lambda row: [
                    highlight_invalid_phone(row["PhoneNumber"], row["PhoneNumber_Valid_Flag"])
                    if col == "PhoneNumber" else ''
                    for col in df.columns
                ],
                axis=1
            )

            st.dataframe(styled_df)
        else:
            st.warning("No PhoneNumber column found in uploaded file.")

        # ---- Detected Issues Summary ----
        st.subheader("Detected Issues")
        invalid_count = (~df["PhoneNumber_Valid_Flag"]).sum() if "PhoneNumber_Valid_Flag" in df else 0
        st.write(f"Invalid Phone Numbers: {invalid_count}")

    except Exception as e:
        st.error(f"Error reading file: {e}")
