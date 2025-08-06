import numpy as np
import streamlit as st
import pandas as pd
import tempfile
import os
import json
from Import_file import DataIngestionBackend

# Page config
st.set_page_config(
    page_title="Data Ingestion Backend Tester",
    page_icon="📊",
    layout="wide"
)

# Initialize backend
@st.cache_resource
def get_backend():
    return DataIngestionBackend()

backend = get_backend()

st.title("📊 Data Ingestion Backend Tester")
st.markdown("Test the data ingestion backend with various file types and API endpoints.")

# Sidebar for navigation
st.sidebar.title("🔧 Options")
data_source = st.sidebar.selectbox(
    "Select Data Source",
    ["File Upload", "API Endpoint"]
)

if data_source == "File Upload":
    st.header("📁 File Upload Testing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json', 'xml'],
        help="Supported formats: CSV, Excel (XLSX/XLS), JSON, XML"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        try:
            # Get file info
            file_info = backend.get_file_info(file_path)
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{file_info['file_size_bytes'] / 1024:.1f} KB")
            with col2:
                st.metric("File Type", file_info['file_extension'].upper())
            with col3:
                st.metric("Supported", "✅" if file_info['supported'] else "❌")
            
            # File-specific options
            file_ext = file_info['file_extension']
            
            # Common options
            st.subheader("🔧 Processing Options")
            
            col1, col2 = st.columns(2)
            
            # CSV/Excel specific options
            if file_ext in ['csv', 'xlsx', 'xls']:
                with col1:
                    st.write("**Row Trimming Options**")
                    top_rows = st.number_input("Remove top rows", min_value=0, max_value=50, value=0)
                    bottom_rows = st.number_input("Remove bottom rows", min_value=0, max_value=50, value=0)
                
                with col2:
                    st.write("**CSV Options** (CSV only)")
                    if file_ext == 'csv':
                        delimiter = st.selectbox("Delimiter", [',', ';', '\t', '|'], index=0)
                        encoding = st.selectbox("Encoding", ['utf-8', 'latin-1', 'cp1252'], index=0)
                    else:
                        delimiter = ','
                        encoding = 'utf-8'
            
            # JSON specific options
            elif file_ext == 'json':
                with col1:
                    st.write("**JSON Options**")
                    flatten_json = st.checkbox("Flatten nested JSON", value=False)
                    extract_path = st.text_input(
                        "Extract path (e.g., 'data.items')", 
                        value="",
                        help="Leave empty to use root level"
                    )
                
                top_rows = bottom_rows = 0
                delimiter = ','
                encoding = 'utf-8'
            
            # XML specific options
            elif file_ext == 'xml':
                with col1:
                    st.write("**XML Parsing Options**")
                    parsing_method = st.selectbox(
                        "Parsing Method",
                        ['elementtree', 'dom', 'sax', 'iterparse', 'xpath'],
                        help="Different XML parsing approaches"
                    )
                    flatten_xml = st.checkbox("Flatten nested XML", value=True, help="Recommended for better DataFrame structure")
                
                with col2:
                    st.write("**Advanced Options**")
                    if parsing_method == 'xpath':
                        xpath_query = st.text_input(
                            "XPath Query", 
                            value="//record",
                            help="XPath expression for targeted extraction (e.g., //employee, //item)"
                        )
                    else:
                        xpath_query = None
                        st.info("💡 For complex XML, try 'xpath' method with specific queries")
                
                top_rows = bottom_rows = 0
                delimiter = ','
                encoding = 'utf-8'
            
            # Load and process data
            if st.button("🚀 Load and Process Data", type="primary"):
                with st.spinner("Loading data..."):
                    try:
                        # Load data based on file type
                        if file_ext == 'csv':
                            df = backend.load_csv(
                                file_path, 
                                top_rows=top_rows, 
                                bottom_rows=bottom_rows,
                                delimiter=delimiter,
                                encoding=encoding
                            )
                        elif file_ext in ['xlsx', 'xls']:
                            df = backend.load_excel(
                                file_path, 
                                top_rows=top_rows, 
                                bottom_rows=bottom_rows
                            )
                        elif file_ext == 'json':
                            df = backend.load_json(
                                file_path, 
                                flatten=flatten_json,
                                extract_path=extract_path if extract_path else None
                            )
                        elif file_ext == 'xml':
                            df = backend.load_xml(
                                file_path,
                                parsing_method=parsing_method,
                                flatten=flatten_xml,
                                xpath_query=xpath_query
                            )
                        
                        st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                        
                        # Store in session state
                        st.session_state.df = df
                        st.session_state.original_df = df.copy()
                        
                    except Exception as e:
                        st.error(f"❌ Error loading data: {str(e)}")
        
        finally:
            # Clean up temp file
            if os.path.exists(file_path):
                os.unlink(file_path)

else:  # API Endpoint
    st.header("🌐 API Endpoint Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**API Configuration**")
        api_url = st.text_input(
            "API URL", 
            value="https://jsonplaceholder.typicode.com/posts",
            help="Enter the API endpoint URL"
        )
        http_method = st.selectbox("HTTP Method", ['GET', 'POST'])
        data_format = st.selectbox("Response Format", ['json', 'xml'])
    
    with col2:
        st.write("**Optional Parameters**")
        
        # Headers
        st.write("Headers (JSON format):")
        headers_text = st.text_area(
            "Headers",
            value='{"Content-Type": "application/json"}',
            height=100,
            label_visibility="collapsed"
        )
        
        # Parameters
        st.write("Parameters (JSON format):")
        params_text = st.text_area(
            "Parameters",
            value='{}',
            height=100,
            label_visibility="collapsed"
        )
    
    # Processing options
    col1, col2 = st.columns(2)
    with col1:
        extract_path_api = st.text_input(
            "Extract path (e.g., 'data.results')", 
            value="",
            help="Leave empty to use root level"
        )
    with col2:
        flatten_api = st.checkbox("Flatten nested data", value=False)
    
    if st.button("🚀 Fetch API Data", type="primary"):
        with st.spinner("Fetching data from API..."):
            try:
                # Parse headers and params
                headers = json.loads(headers_text) if headers_text.strip() else None
                params = json.loads(params_text) if params_text.strip() else None
                
                # Fetch data
                df = backend.load_from_api(
                    url=api_url,
                    headers=headers,
                    params=params,
                    method=http_method,
                    data_format=data_format,
                    extract_path=extract_path_api if extract_path_api else None,
                    flatten=flatten_api
                )
                
                st.success(f"✅ API data loaded successfully! Shape: {df.shape}")
                
                # Store in session state
                st.session_state.df = df
                st.session_state.original_df = df.copy()
                
            except Exception as e:
                st.error(f"❌ Error fetching API data: {str(e)}")

# Display and process loaded data
if 'df' in st.session_state:
    st.header("📊 Data Preview and Processing")
    
    df = st.session_state.df
    
    # Data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    with col4:
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("Null Values", f"{null_percentage:.1f}%")
    
    # Show data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(100), use_container_width=True, height=300)
    
    # Data processing options
    st.subheader("⚙️ Data Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Date Column Processing**")
        
        # Detect date columns
        if st.button("🔍 Detect Date Columns"):
            with st.spinner("Detecting date columns..."):
                detected_dates = backend.detect_date_columns(df)

                # Normalize detected_dates to the same type as df.columns if needed
                current_columns = list(df.columns)
                if all(isinstance(c, str) for c in current_columns):
                    # ensure everything is string for comparison
                    detected_dates = [str(d) for d in (detected_dates or [])]

                # Keep only detected columns that still exist in current df
                options_set = set(current_columns)
                filtered_detected = [d for d in (detected_dates or []) if d in options_set]

                # Save filtered (valid) defaults in session state
                st.session_state.detected_dates = filtered_detected

                if filtered_detected:
                    st.success(f"Found {len(filtered_detected)} potential date columns: {', '.join(filtered_detected)}")
                else:
                    st.info("No date columns detected")

        # Date column selection
        options = df.columns.tolist()
        detected = st.session_state.get('detected_dates', []) or []

        # If columns are strings, ensure detected defaults are strings too
        if all(isinstance(c, str) for c in options):
            detected = [str(d) for d in detected]

        # Filter defaults so Streamlit won't raise an exception for missing defaults
        options_set = set(options)
        filtered_default = [d for d in detected if d in options_set]

        # Optional: show info if some stale defaults were dropped
        dropped = [d for d in detected if d not in options_set]
        if dropped:
            st.info(f"Removed {len(dropped)} stale detected defaults: {', '.join(map(str, dropped))}")

        date_columns_to_convert = st.multiselect(
            "Select columns to convert to dates",
            options=options,
            default=filtered_default
        )

        date_format = st.selectbox(
            "Target date format",
            ['dd/mm/yyyy'],
            help="Currently only dd/mm/yyyy is supported"
        )

    with col2:
        st.write("**Data Cleaning Options**")
        
        # Show current data quality info
        if df is not None:
            empty_rows = int(df.isnull().all(axis=1).sum())
            empty_cols = int(df.isnull().all().sum())

            st.info(f"🧹 Empty rows (will be removed automatically): {empty_rows}")
            
            # User choice for empty columns
            remove_empty_columns = st.checkbox(
                f"Remove empty columns ({empty_cols} detected)", 
                value=False,
                help="Check this to remove completely empty columns. Empty rows are always removed."
            )

            # Additional info
            if empty_rows > 0:
                st.caption("✅ Empty rows will be automatically removed when processing is applied.")
            else:
                st.caption("✅ No completely empty rows detected.")

            if empty_cols > 0:
                if remove_empty_columns:
                    st.caption("✅ Empty columns will be removed during processing.")
                else:
                    st.caption("ℹ️ Empty columns will be kept (uncheck to remove them).")
            else:
                st.caption("✅ No completely empty columns detected.")
    
    # Process data
    if st.button("⚡ Apply Processing", type="primary"):
        with st.spinner("Processing data..."):
            try:
                processed_df = backend.process_data(
                    df=st.session_state.original_df.copy(),
                    date_columns=date_columns_to_convert,
                    date_format=date_format,
                    remove_empty_columns=remove_empty_columns,  # Now this parameter exists
                    force_date_conversion=True
                )

                # Save processed results & flags to session state
                st.session_state.df = processed_df
                st.session_state.processing_applied = True
                st.session_state.date_columns_converted = date_columns_to_convert
                st.session_state.empty_columns_removed = remove_empty_columns

                st.success("✅ Data processed successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")

# Final Processing Results View
if 'df' in st.session_state and st.session_state.get('processing_applied', False):
    st.header("🎯 Final Processing Results")
    
    # Processing Summary
    st.subheader("📋 Processing Summary")
    
    original_df = st.session_state.original_df
    processed_df = st.session_state.df
    
    # Create summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rows_before = len(original_df)
        rows_after = len(processed_df)
        rows_removed = rows_before - rows_after
        st.metric("Rows", f"{rows_after:,}", delta=f"{-rows_removed:,}" if rows_removed != 0 else "0")
    
    with col2:
        cols_before = len(original_df.columns)
        cols_after = len(processed_df.columns)
        cols_removed = cols_before - cols_after
        st.metric("Columns", f"{cols_after}", delta=f"{-cols_removed}" if cols_removed != 0 else "0")
    
    with col3:
        null_before = original_df.isnull().sum().sum()
        null_after = processed_df.isnull().sum().sum()
        null_removed = null_before - null_after
        st.metric("Null Values", f"{null_after:,}", delta=f"{-null_removed:,}" if null_removed != 0 else "0")
    
    with col4:
        memory_before = original_df.memory_usage(deep=True).sum() / 1024
        memory_after = processed_df.memory_usage(deep=True).sum() / 1024
        memory_diff = memory_before - memory_after
        st.metric("Memory (KB)", f"{memory_after:.1f}", delta=f"{-memory_diff:.1f}" if memory_diff > 0.1 else "0")
    
    # Processing Details
    st.subheader("🔧 Applied Transformations")
    
    transformations = []
    
    # Always mention empty row removal since it's automatic
    empty_rows_removed = len(original_df) - len(original_df.dropna(how='all'))
    if empty_rows_removed > 0:
        transformations.append(f"🧹 **Empty Row Removal**: Automatically removed {empty_rows_removed} completely empty rows")
    
    # Empty column removal (only if user opted for it)
    if st.session_state.get('empty_columns_removed', False):
        empty_cols_removed = len(original_df.columns) - len(original_df.dropna(axis=1, how='all').columns)
        if empty_cols_removed > 0:
            transformations.append(f"🗑️ **Empty Column Removal**: Removed {empty_cols_removed} completely empty columns")
    
    # Date conversions
    if st.session_state.get('date_columns_converted'):
        date_cols = st.session_state.date_columns_converted
        transformations.append(f"📅 **Date Conversion**: Converted {len(date_cols)} columns to dd/mm/yyyy format: {', '.join(date_cols)}")
    
    # Data type changes
    original_dtypes = original_df.dtypes.value_counts()
    processed_dtypes = processed_df.dtypes.value_counts()
    
    if not original_dtypes.equals(processed_dtypes):
        transformations.append("🔄 **Data Type Changes**: Column data types were modified during processing")
    
    if transformations:
        for transformation in transformations:
            st.markdown(transformation)
    else:
        st.info("ℹ️ No transformations were applied to the data")
    
    # Data Quality Comparison
    st.subheader("📊 Data Quality Comparison")
    
    # Create comparison DataFrame
    quality_metrics = {
        'Metric': ['Total Rows', 'Total Columns', 'Null Values', 'Memory Usage (KB)', 'Duplicate Rows'],
        'Original Data': [
            f"{len(original_df):,}",
            f"{len(original_df.columns)}",
            f"{original_df.isnull().sum().sum():,}",
            f"{original_df.memory_usage(deep=True).sum() / 1024:.1f}",
            f"{original_df.duplicated().sum():,}"
        ],
        'Processed Data': [
            f"{len(processed_df):,}",
            f"{len(processed_df.columns)}",
            f"{processed_df.isnull().sum().sum():,}",
            f"{processed_df.memory_usage(deep=True).sum() / 1024:.1f}",
            f"{processed_df.duplicated().sum():,}"
        ]
    }
    
    quality_df = pd.DataFrame(quality_metrics)
    st.dataframe(quality_df, hide_index=True, use_container_width=True)
    
    # Column Details Comparison (if columns changed)
    if len(original_df.columns) != len(processed_df.columns) or not all(original_df.columns == processed_df.columns):
        st.subheader("📋 Column Changes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Columns**")
            original_cols_info = pd.DataFrame({
                'Column': original_df.columns,
                'Data Type': [str(dtype) for dtype in original_df.dtypes],
                'Non-Null Count': [original_df[col].count() for col in original_df.columns]
            })
            st.dataframe(original_cols_info, hide_index=True, height=300)
        
        with col2:
            st.write("**Processed Columns**")
            processed_cols_info = pd.DataFrame({
                'Column': processed_df.columns,
                'Data Type': [str(dtype) for dtype in processed_df.dtypes],
                'Non-Null Count': [processed_df[col].count() for col in processed_df.columns]
            })
            st.dataframe(processed_cols_info, hide_index=True, height=300)
    
    # Final Data Preview
    st.subheader("👁️ Final Processed Data")
    
    # Show data with pagination
    # Handle small DataFrames and avoid slider errors
    max_rows = len(processed_df)
    slider_min = 1
    slider_max = min(1000, max_rows)
    default_value = min(100, max_rows)

    if slider_max == slider_min:
        # No range for slider — just show all rows
        show_rows = slider_max
    else:
        step = 10 if (slider_max - slider_min) >= 10 else 1
        show_rows = st.slider(
            "Number of rows to display",
            min_value=slider_min,
            max_value=slider_max,
            value=default_value,
            step=step
        )

    if show_rows < len(processed_df):
        st.info(f"Showing first {show_rows} rows of {len(processed_df):,} total rows")

    # Prepare display DataFrame and format date columns as dd/mm/yyyy for UI
    display_df = processed_df.copy()
    if st.session_state.get('date_columns_converted'):
        for col in st.session_state.date_columns_converted:
            if col in display_df.columns:
                # Check if the column contains dates that need formatting
                try:
                    display_df[col] = pd.to_datetime(display_df[col], errors='coerce').dt.strftime('%d/%m/%Y')
                    display_df[col] = display_df[col].fillna('')
                except:
                    # If formatting fails, keep original values
                    pass

    st.dataframe(display_df.head(show_rows), use_container_width=True, height=400)

    # Sample Statistics (for numeric columns)
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.subheader("📈 Numeric Data Statistics")
        
        with st.expander("View Statistics", expanded=False):
            stats_df = processed_df[numeric_cols].describe()
            st.dataframe(stats_df, use_container_width=True)
    
    # Data Validation Summary
    st.subheader("✅ Data Validation Summary")
    
    validation_results = []
    
    # Check for common data issues
    if processed_df.empty:
        validation_results.append("❌ **Critical**: DataFrame is empty after processing")
    else:
        validation_results.append("✅ **Good**: DataFrame contains data")
    
    # Check for null values
    null_percentage = (processed_df.isnull().sum().sum() / (len(processed_df) * len(processed_df.columns)) * 100)
    if null_percentage == 0:
        validation_results.append("✅ **Excellent**: No null values in the dataset")
    elif null_percentage < 5:
        validation_results.append(f"⚠️ **Good**: Low null value percentage ({null_percentage:.1f}%)")
    elif null_percentage < 20:
        validation_results.append(f"⚠️ **Moderate**: Some null values present ({null_percentage:.1f}%)")
    else:
        validation_results.append(f"❌ **Attention**: High null value percentage ({null_percentage:.1f}%)")
    
    # Check for duplicates
    duplicate_count = processed_df.duplicated().sum()
    if duplicate_count == 0:
        validation_results.append("✅ **Good**: No duplicate rows found")
    else:
        validation_results.append(f"⚠️ **Notice**: {duplicate_count:,} duplicate rows found")
    
    # Check date columns if any were converted
    if st.session_state.get('date_columns_converted'):
        date_conversion_issues = 0
        for col in st.session_state.date_columns_converted:
            if col in processed_df.columns:
                # Check if conversion was successful
                try:
                    pd.to_datetime(processed_df[col], format='%d/%m/%Y', errors='raise')
                except:
                    date_conversion_issues += 1
        
        if date_conversion_issues == 0:
            validation_results.append("✅ **Good**: All date conversions successful")
        else:
            validation_results.append(f"⚠️ **Notice**: {date_conversion_issues} date columns may have conversion issues")
    
    for result in validation_results:
        st.markdown(result)
    
    # Data export
    st.subheader("💾 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download CSV"):
            csv = processed_df.to_csv(index=False)
            st.download_button(
                label="💾 Save CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📥 Download Excel"):
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                processed_df.to_excel(writer, sheet_name='Data', index=False)
            
            st.download_button(
                label="💾 Save Excel",
                data=buffer.getvalue(),
                file_name="processed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col3:
        if st.button("📥 Download JSON"):
            json_data = processed_df.to_json(orient='records', indent=2)
            st.download_button(
                label="💾 Save JSON",
                data=json_data,
                file_name="processed_data.json",
                mime="application/json"
            )
    
    # Reset data
    if st.button("🔄 Reset to Original Data", help="Reset to the originally loaded data"):
        if 'original_df' in st.session_state:
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.processing_applied = False  # Reset processing flag
            st.success("✅ Data reset to original!")
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚀 Data Ingestion Backend Tester | Built with Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)