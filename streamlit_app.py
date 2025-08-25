import streamlit as st
import pandas as pd
from utils.phone_validator import is_phone_number_valid
from utils.blood_sugar_validator import is_blood_sugar_valid
# AgGrid imports
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

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

        # Add validity columns using Python validation functions only if columns exist
        has_phone = "PhoneNumber" in df.columns
        has_blood = "BloodSugar" in df.columns
        if has_phone:
            df["PhoneNumber_Valid"] = df["PhoneNumber"].apply(is_phone_number_valid)
        if has_blood:
            df["BloodSugar_Valid"] = df["BloodSugar"].apply(is_blood_sugar_valid)

        # Count invalid entries
        invalid_phone_count = (~df["PhoneNumber_Valid"]).sum() if has_phone else None
        invalid_blood_sugar_count = (~df["BloodSugar_Valid"]).sum() if has_blood else None

        # AgGrid cell style JS using validity columns only if columns exist
        cell_style_jscode = JsCode('''
        function(params) {
            if (params.colDef.field === 'PhoneNumber' && params.data.PhoneNumber_Valid === false) {
                return { 'backgroundColor': '#2b0000' };
            }
            if (params.colDef.field === 'BloodSugar' && params.data.BloodSugar_Valid === false) {
                return { 'backgroundColor': '#2b0000' };
            }
            return { 'backgroundColor': '#181818' };
        }
        ''')

        # Hide validity columns from display but keep for validation
        display_columns = [col for col in df.columns if not col.endswith('_Valid')]
        gb = GridOptionsBuilder.from_dataframe(df[display_columns])
        gb.configure_default_column(editable=True, cellStyle=cell_style_jscode)
        gridOptions = gb.build()

        st.subheader("Validated & Editable Data Table")
        grid_response = AgGrid(
            df,
            gridOptions=gridOptions,
            enable_enterprise_modules=False,
            update_mode='MODEL_CHANGED',
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW' # optional: fit columns
        )


        edited_df = grid_response['data'] if 'data' in grid_response else df

        # Recount invalids after edit using validity columns only if columns exist
        has_phone_edit = "PhoneNumber" in edited_df.columns
        has_blood_edit = "BloodSugar" in edited_df.columns
        if has_phone_edit:
            edited_df["PhoneNumber_Valid"] = edited_df["PhoneNumber"].apply(is_phone_number_valid)
        if has_blood_edit:
            edited_df["BloodSugar_Valid"] = edited_df["BloodSugar"].apply(is_blood_sugar_valid)
        invalid_phone_count = (~edited_df["PhoneNumber_Valid"]).sum() if has_phone_edit else None
        invalid_blood_sugar_count = (~edited_df["BloodSugar_Valid"]).sum() if has_blood_edit else None

        st.subheader("Detected Issues")
        if has_phone_edit:
            st.write(f"Invalid phone numbers: **{invalid_phone_count}**")
        else:
            st.warning("No PhoneNumber column found in the uploaded file.")
        if has_blood_edit:
            st.write(f"Invalid blood sugar entries: **{invalid_blood_sugar_count}**")
        else:
            st.warning("No BloodSugar column found in the uploaded file.")

        # Export edited data
        st.subheader("Export Corrected Data")
        export_format = st.selectbox("Choose export format", ["csv", "xlsx"])
        if st.button("Download edited file"):
            if export_format == "csv":
                st.download_button("Download CSV", edited_df.to_csv(index=False), file_name="corrected_data.csv", mime="text/csv")
            else:
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    edited_df.to_excel(writer, index=False)
                st.download_button("Download Excel", output.getvalue(), file_name="corrected_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"Error reading file: {e}")
