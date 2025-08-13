import streamlit as st
import pandas as pd
from Data_validation import DataValidator, quick_data_profile

st.set_page_config(page_title="🧪 Data Validation Tester", layout="wide")
st.title("📊 Data Validation Streamlit UI")

# Upload any file
uploaded_file = st.file_uploader(
    "📁 Upload your dataset", 
    type=["csv", "xlsx", "xls", "json", "parquet", "feather"]
)

if uploaded_file:
    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        elif file_name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        elif file_name.endswith('.feather'):
            df = pd.read_feather(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

    # 🔹 Cast object columns to string for Arrow compatibility
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    st.subheader("🔍 Preview Uploaded Data")
    st.dataframe(df)

    st.markdown("---")
    st.subheader("⚙️ Define Validation Rules")
    columns = list(df.columns)

    # Unique columns
    unique_columns = st.multiselect("✅ Columns that must be unique", options=columns)

    # Composite unique columns
    st.text("🧩 Composite unique sets (e.g. [['id', 'email']])")
    composite_unique_input = st.text_area("Composite Unique", value="[[]]", height=100)
    try:
        composite_unique = eval(composite_unique_input)
        if not isinstance(composite_unique, list):
            composite_unique = []
    except:
        composite_unique = []

    # Non-null columns
    non_null_columns = st.multiselect("🚫 Non-null columns", options=columns)

    # Numeric column rules
    st.markdown("🔢 Numeric Column Rules")
    selected_numeric_columns = st.multiselect("Select numeric columns", options=columns)
    numeric_config = {}
    for col in selected_numeric_columns:
        st.markdown(f"**Column: `{col}`**")
        col_min = st.number_input(f"Minimum value for `{col}`", key=f"min_{col}", value=0)
        col_max = st.number_input(f"Maximum value for `{col}`", key=f"max_{col}", value=1000000)
        numeric_config[col] = {"min": col_min, "max": col_max}

    # Date column rules
    st.markdown("📅 Date Column Rule")
    date_col = st.selectbox("Choose a date column (optional)", options=[""] + columns)
    date_config = {}
    if date_col:
        date_format = st.selectbox("Date format", options=["dd/mm/yyyy", "dd-mm-yyyy"])
        min_date = st.text_input("Minimum date", value="01/01/2000")
        max_date = st.text_input("Maximum date", value="31/12/2030")
        date_config[date_col] = {
            "format": date_format,
            "min_date": min_date,
            "max_date": max_date
        }

    # Categorical columns
    st.markdown("🧬 Categorical Column Rule")
    cat_col = st.selectbox("Choose a categorical column (optional)", options=[""] + columns)
    categorical_config = {}
    if cat_col:
        allowed_vals = st.text_input("Allowed values (comma-separated)", value="Yes,No")
        allowed_list = [v.strip() for v in allowed_vals.split(",")]
        categorical_config[cat_col] = allowed_list

    # Combine validation config
    validation_config = {
        "unique_columns": unique_columns,
        "composite_unique": composite_unique,
        "non_null_columns": non_null_columns,
        "numeric_columns": numeric_config,
        "date_columns": date_config,
        "categorical_columns": categorical_config
    }

    st.markdown("---")
    if st.button("🚦 Run Validation"):
        validator = DataValidator(validation_config)
        result = validator.validate(df)
        summary = result.get_summary()

        if summary["is_valid"]:
            st.success("✅ Data is valid.")
        else:
            st.error("❌ Data validation failed.")

        st.subheader("📋 Validation Summary")
        st.json(summary)

        if summary["errors"]:
            st.subheader("❗ Validation Errors")
            for err in summary["errors"]:
                st.write(f"**{err['error_type']}** in `{err['column']}` — {err['details']}")
                if err['invalid_rows']:
                    st.code(f"Rows: {err['invalid_rows'][:10]}{'...' if len(err['invalid_rows']) > 10 else ''}")

        if summary["warnings"]:
            st.subheader("⚠️ Warnings")
            for warn in summary["warnings"]:
                st.write(f"**{warn['warning_type']}** in `{warn['column']}` — {warn['details']}")

    if st.checkbox("📑 Show Quick Data Profile"):
        profile = quick_data_profile(df)
        st.subheader("📊 Data Profile")
        st.json(profile)

