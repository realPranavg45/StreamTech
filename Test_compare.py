"""
streamlit_compare.py

A Streamlit test UI for the data_diff backend.
Drop in same folder as data_diff.py and run:
    streamlit run streamlit_compare.py
"""

import streamlit as st
import pandas as pd
from Compare import compare_dataframes, make_diff_styler, make_diff_html, df_to_csv_bytes,split_old_new_tables

st.set_page_config(page_title="Data Compare Tester", layout="wide")

st.title("Dataset Compare Tester")

st.markdown("""
Upload **old** and **new** CSV files (or use the sample datasets).  
Select the primary key column(s) and click **Compare**.
""")

col1, col2 = st.columns(2)

with col1:
    uploaded_old = st.file_uploader("Upload OLD dataset (CSV)", type=['csv'], key='old')
    use_sample = st.checkbox("Use sample datasets (instead of upload)", value=False, key='sample')

with col2:
    uploaded_new = st.file_uploader("Upload NEW dataset (CSV)", type=['csv'], key='new')

def load_df_from_uploaded(uploaded):
    if uploaded is None:
        return None
    try:
        return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

@st.cache_data
def sample_data():
    df_old = pd.DataFrame([
        {'id': 1, 'name': 'Alice', 'score': 90, 'department': 'Sales'},
        {'id': 2, 'name': 'Bob', 'score': 75, 'department': 'Support'},
        {'id': 3, 'name': 'Carl', 'score': 88, 'department': 'Engineering'},
    ])
    df_new = pd.DataFrame([
        {'id': 1, 'name': 'Alice', 'score': 91, 'department': 'Sales'},  # changed score
        {'id': 2, 'name': 'Bob', 'score': 75, 'department': 'Customer Success'},  # changed department
        {'id': 4, 'name': 'Diana', 'score': 82, 'department': 'Support'},  # new row
    ])
    return df_old, df_new

if use_sample:
    old_df, new_df = sample_data()
else:
    old_df = load_df_from_uploaded(uploaded_old)
    new_df = load_df_from_uploaded(uploaded_new)

if old_df is None or new_df is None:
    st.info("Upload both files or enable sample datasets.")
    st.stop()

st.subheader("Preview (first 5 rows)")
c1, c2 = st.columns(2)
with c1:
    st.write("**OLD**")
    st.dataframe(old_df.head())
with c2:
    st.write("**NEW**")
    st.dataframe(new_df.head())

# Choose PK columns (intersection recommended)
common_cols = sorted(list(set(old_df.columns) & set(new_df.columns)))
st.write("Select primary key column(s) — must be present in both datasets.")
pk_cols = st.multiselect("Primary key columns", options=common_cols, default=[common_cols[0]] if common_cols else [])

if not pk_cols:
    st.warning("Choose at least one primary key column to compare.")
    st.stop()

if st.button("Compare"):
    try:
        results = compare_dataframes(old_df, new_df, pk_cols)
    except Exception as e:
        st.error(f"Comparison failed: {e}")
        st.stop()

    new_rows = results['new_rows']
    deleted_rows = results['deleted_rows']
    modified_merged = results['modified_merged']

    st.success("Comparison complete.")
    st.markdown(f"**Summary:** New rows = {len(new_rows)}, Deleted rows = {len(deleted_rows)}, Modified rows = {len(modified_merged)}")

    tab1, tab2, tab3 = st.tabs(["New rows", "Deleted rows", "Modified rows"])

    def is_effectively_empty(df):
        if df is None:
            return True
        if getattr(df, "shape", (0,))[0] == 0:
            return True
        # if every row is all-NA, treat as empty
        return df.dropna(how="all").shape[0] == 0

    with tab1:
        if is_effectively_empty(new_rows):
            st.info("No new rows.")
        else:
            st.dataframe(new_rows)
            b = df_to_csv_bytes(new_rows)
            st.download_button("Download new rows CSV", data=b, file_name="new_rows.csv")

    with tab2:
        if deleted_rows.empty:
            st.info("No deleted rows.")
        else:
            st.dataframe(deleted_rows)
            b = df_to_csv_bytes(deleted_rows)
            st.download_button("Download deleted rows CSV", data=b, file_name="deleted_rows.csv")

    with tab3:
        if modified_merged.empty:
            st.info("No modified rows.")
        else:
            st.write("### Modified rows — Old vs New")

            # Split into old & new DataFrames
            old_df, new_df = split_old_new_tables(modified_merged, pk_cols)

            # Apply diff highlighting only on changed cells
            try:
                styler_old = make_diff_styler(modified_merged, pk_cols, view="old")
                styler_new = make_diff_styler(modified_merged, pk_cols, view="new")
            except TypeError:
                # If your make_diff_styler doesn't support view arg
                styler_old = old_df.style
                styler_new = new_df.style

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Old Table**")
                st.table(make_diff_styler(modified_merged, pk_cols, view="old"))  # or st.markdown(styler_old.to_html(), unsafe_allow_html=True)
            with col2:
                st.markdown("**New Table**")
                st.table(make_diff_styler(modified_merged, pk_cols, view="new")) # or st.markdown(styler_new.to_html(), unsafe_allow_html=True)


            # Downloads
            b = df_to_csv_bytes(modified_merged)
            st.download_button("Download modified (merged) CSV", data=b, file_name="modified_merged.csv")
