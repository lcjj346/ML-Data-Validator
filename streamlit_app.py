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

        # Generic validation: mark empty or non-numeric cells as invalid (customize as needed)
        def generic_validator(val):
            if pd.isnull(val):
                return False
            # Try to convert to float, if fails, mark as invalid
            try:
                float(val)
                return True
            except:
                return False

        # Add validity columns for all columns except those ending with _Valid
        original_columns = [col for col in df.columns if not col.endswith('_Valid')]
        for col in original_columns:
            if col == "PhoneNumber":
                df[f"{col}_Valid"] = df[col].apply(is_phone_number_valid)
            else:
                df[f"{col}_Valid"] = df[col].apply(generic_validator)

        # Count invalid entries for all columns except those ending with _Valid
        invalid_counts = {col: (~df[f"{col}_Valid"]).sum() for col in original_columns}

        # AgGrid cell style JS for all columns
        cell_style_jscode = JsCode('''
        function(params) {
            if (params.data[params.colDef.field + '_Valid'] === false) {
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

        # Re-validate all columns after edit
        for col in display_columns:
            if not col.endswith('_Valid'):
                if col == "PhoneNumber":
                    edited_df[f"{col}_Valid"] = edited_df[col].apply(is_phone_number_valid)
                else:
                    edited_df[f"{col}_Valid"] = edited_df[col].apply(generic_validator)
        invalid_counts = {col: (~edited_df[f"{col}_Valid"]).sum() for col in display_columns if not col.endswith('_Valid')}

        st.subheader("Detected Issues")
        for col, count in invalid_counts.items():
            st.write(f"Invalid entries in {col}: **{count}**")

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
