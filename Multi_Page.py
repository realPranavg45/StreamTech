# multi.py
import streamlit as st

# ---------------------
# Page Navigation Setup
# ---------------------
pg = st.navigation([
    st.Page("Test_compare.py", title="🔍 Compare Data"),
    st.Page("Test_dataframe.py", title="📊 Dataframe Viewer"),
    st.Page("Test_delete.py", title="🗑️ Delete Records"),
    st.Page("Test_email_backend.py", title="📧 Email Backend"),
    st.Page("Test_get_data.py", title="📥 Fetch Data"),
    st.Page("Test_mapper.py", title="🔄 Column Mapper"),
    st.Page("Test_Data_Validation.py", title="🧪 Data Validation"),
    st.Page("Test_google_api.py", title="🌐 Google API Integration"),
    st.Page("Test_import_file.py", title="📂 Import Files"),
    st.Page("Test_Data_Reconciliation.py", title="🔗⚖️ Data Reconciliation")
    st.Page("Test_update.py", title="✏️ Update Records"),
])


pg.run()




