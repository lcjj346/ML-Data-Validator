import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid.shared import JsCode

# Validation functions
def is_phone_number_valid(x):
    return str(x).isdigit() and len(str(x)) == 10

def is_blood_sugar_valid(x):
    try:
        val = float(x)
        return 70 <= val <= 200
    except:
        return False

st.title("Editable & Highlighted Table with Export (using AG Grid)")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Add a JS function to style invalid cells
    cell_style_jscode = JsCode("""
    function(params) {
        if (params.colDef.field === 'PhoneNumber') {
            if (!params.value || !/^\\d{10}$/.test(params.value)) {
                return { 'backgroundColor': 'red', 'color': 'white' }
            }
        }
        if (params.colDef.field === 'BloodSugar') {
            let val = parseFloat(params.value);
            if (isNaN(val) || val < 70 || val > 200) {
                return { 'backgroundColor': 'red', 'color': 'white' }
            }
        }
        return null;
    };
    """)

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=True, cellStyle=cell_style_jscode)
    grid_options = gb.build()

    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
    )

    edited_df = pd.DataFrame(grid_response['data'])

    st.subheader("Export Edited Data")
    csv = edited_df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Edited CSV", data=csv, file_name="edited_data.csv", mime="text/csv")
