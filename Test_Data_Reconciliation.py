import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO, BytesIO
import sys
import os
import json
import pyarrow.parquet as pq
import tempfile
from typing import Dict, List, Tuple, Union

# Add the current directory to path to import the reconciliation engine
# Make sure the Data_Reconciliation.py file is in the same directory
try:
    from Data_Reconciliation import DataReconciliationEngine
except ImportError:
    st.error("Please make sure Data_Reconciliation.py is in the same directory as this script")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Universal Data Reconciliation Engine",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Universal Data Reconciliation Engine")
st.markdown("""
This app tests the enhanced Data Reconciliation Engine with features like:
- Support for multiple file formats (CSV, Excel, JSON, Parquet)
- Exact matching with comma-separated indexes
- Duplicate key combination
- Partial matching with configurable thresholds
- Comprehensive final summary DataFrame
- Manual reconciliation capability
""")

# Supported file formats
SUPPORTED_FORMATS = {
    'csv': {'extensions': ['csv', 'txt'], 'reader': pd.read_csv, 'writer': 'to_csv'},
    'excel': {'extensions': ['xls', 'xlsx', 'xlsm'], 'reader': pd.read_excel, 'writer': 'to_excel'},
    'json': {'extensions': ['json'], 'reader': pd.read_json, 'writer': 'to_json'},
    'parquet': {'extensions': ['parquet'], 'reader': pd.read_parquet, 'writer': 'to_parquet'}
}

# Sidebar for configuration
st.sidebar.header("Configuration")

# Sample data option
use_sample_data = st.sidebar.checkbox("Use Sample Data", value=False)

if use_sample_data:
    st.sidebar.markdown("### Sample Data Loaded")
    st.sidebar.info("Using pre-built sample datasets with duplicates to demonstrate all features")
    left_name = "Sample Left Dataset"
    right_name = "Sample Right Dataset"
else:
    st.sidebar.markdown("### Upload Your Data")
    left_file = st.sidebar.file_uploader(
        "Upload Left Dataset", 
        type=list(set(ext for fmt in SUPPORTED_FORMATS.values() for ext in fmt['extensions']))
    )
    right_file = st.sidebar.file_uploader(
        "Upload Right Dataset", 
        type=list(set(ext for fmt in SUPPORTED_FORMATS.values() for ext in fmt['extensions']))
    )
    left_name = left_file.name if left_file else "Left Dataset"
    right_name = right_file.name if right_file else "Right Dataset"

# Configuration parameters
st.sidebar.header("Reconciliation Parameters")
partial_threshold = st.sidebar.slider("Partial Match Threshold", 0.1, 1.0, 0.8, 0.1)
combine_duplicates = st.sidebar.checkbox("Combine Duplicate Keys", value=True)
show_detailed_results = st.sidebar.checkbox("Show Detailed Results", value=True)

# Main content area
col1, col2 = st.columns(2)

def detect_file_format(uploaded_file):
    """Detect file format based on extension"""
    if uploaded_file is None:
        return None
        
    file_extension = uploaded_file.name.split('.')[-1].lower()
    for fmt, properties in SUPPORTED_FORMATS.items():
        if file_extension in properties['extensions']:
            return fmt
    return None

def read_uploaded_file(uploaded_file):
    """Read uploaded file based on its format"""
    if uploaded_file is None:
        return None
        
    file_format = detect_file_format(uploaded_file)
    if file_format is None:
        st.error(f"Unsupported file format for {uploaded_file.name}")
        return None
    
    try:
        reader = SUPPORTED_FORMATS[file_format]['reader']
        
        # For Excel files, we need to read the bytes
        if file_format == 'excel':
            return reader(uploaded_file)
        # For Parquet files, we need to save to temp file first
        elif file_format == 'parquet':
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            df = reader(tmp_path)
            os.unlink(tmp_path)
            return df
        # For CSV, JSON - we can read directly from bytes
        else:
            return reader(BytesIO(uploaded_file.getvalue()))
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name} as {file_format}: {str(e)}")
        return None

# Function to create sample data
@st.cache_data
def create_sample_data():
    """Create sample datasets with various reconciliation scenarios"""
    
    # Left dataset with duplicates
    left_data = {
        'ID': ['A001', 'A001', 'A002', 'A003', 'A004', 'A005'],
        'Product_Name': ['Widget A', 'Widget A', 'Widget B', 'Widget C', 'Widget D', 'Widget E'],
        'Amount': [100, 50, 200, 150, 75, 120],
        'Quantity': [10, 5, 20, 15, 8, 12],
        'Category': ['Electronics', 'Electronics', 'Tools', 'Electronics', 'Tools', 'Books']
    }
    
    # Right dataset with duplicates and some differences
    right_data = {
        'Product_ID': ['A001', 'A001', 'A002', 'A003', 'A006', 'A007'],
        'Product_Description': ['Widget A', 'Widget A', 'Widget B', 'Widget C', 'Widget F', 'Widget G'],
        'Value': [120, 30, 200, 140, 90, 80],  # A001 total = 150 vs left 150
        'Count': [12, 3, 20, 16, 9, 8],        # A001 total = 15 vs left 15
        'Type': ['Electronics', 'Electronics', 'Tools', 'Electronics', 'Books', 'Tools']
    }
    
    return pd.DataFrame(left_data), pd.DataFrame(right_data)

# Load data
if use_sample_data:
    left_df, right_df = create_sample_data()
else:
    if left_file and right_file:
        left_df = read_uploaded_file(left_file)
        right_df = read_uploaded_file(right_file)
        
        if left_df is None or right_df is None:
            st.error("Failed to load one or both datasets. Please check the file formats.")
            st.stop()
    else:
        st.warning("Please upload both datasets or use sample data")
        st.stop()

# Display input datasets
with col1:
    st.header(f"{left_name}")
    st.dataframe(left_df, use_container_width=True)
    st.write(f"**Shape:** {left_df.shape[0]} rows × {left_df.shape[1]} columns")

with col2:
    st.header(f"{right_name}")
    st.dataframe(right_df, use_container_width=True)
    st.write(f"**Shape:** {right_df.shape[0]} rows × {right_df.shape[1]} columns")

# Column selection
st.header("Column Mapping")
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{left_name} Columns")
    left_key_cols = st.multiselect(
        f"Select Key Columns ({left_name})",
        options=left_df.columns.tolist(),
        default=['ID'] if use_sample_data else []
    )
    left_value_cols = st.multiselect(
        f"Select Value Columns ({left_name})",
        options=[col for col in left_df.columns if col not in left_key_cols],
        default=['Amount', 'Quantity'] if use_sample_data else []
    )

with col2:
    st.subheader(f"{right_name} Columns")
    right_key_cols = st.multiselect(
        f"Select Key Columns ({right_name})",
        options=right_df.columns.tolist(),
        default=['Product_ID'] if use_sample_data else []
    )
    right_value_cols = st.multiselect(
        f"Select Value Columns ({right_name})",
        options=[col for col in right_df.columns if col not in right_key_cols],
        default=['Value', 'Count'] if use_sample_data else []
    )

# Sort both datasets by the selected key columns (if keys are pre-selected)
if left_key_cols:
    left_df = left_df.sort_values(by=left_key_cols).reset_index(drop=True)
if right_key_cols:
    right_df = right_df.sort_values(by=right_key_cols).reset_index(drop=True)

# Validation
if len(left_key_cols) != len(right_key_cols):
    st.error("Number of key columns must match between left and right datasets")
elif len(left_value_cols) != len(right_value_cols):
    st.error("Number of value columns must match between left and right datasets")
elif not (left_key_cols and right_key_cols and left_value_cols and right_value_cols):
    st.warning("Please select at least one key column and one value column for each dataset")
else:
    # Initialize session state for manual reconciliation
    if 'reconciliation_results' not in st.session_state:
        st.session_state.reconciliation_results = None
    if 'engine_initialized' not in st.session_state:
        st.session_state.engine_initialized = False
    
    # Manual Reconciliation Section
    st.header("🔧 Manual Reconciliation")
    
    # Create two columns for the manual reconciliation interface
    manual_col1, manual_col2 = st.columns(2)
    
    with manual_col1:
        st.subheader(f"Select {left_name} Rows")
        # Show the dataframe with row indices visible
        st.dataframe(left_df, use_container_width=True)
        
        # Multi-select for row indices
        max_left_idx = len(left_df) - 1
        selected_left = st.multiselect(
            f"Select row indices from {left_name} (0-{max_left_idx})",
            options=list(range(len(left_df))),
            format_func=lambda x: f"Row {x}: {left_df.loc[x, left_key_cols[0]] if len(left_key_cols) > 0 else f'Index {x}'}"
        )
        
        # Display selected rows
        if selected_left:
            st.write("Selected rows:")
            st.dataframe(left_df.loc[selected_left], use_container_width=True)
    
    with manual_col2:
        st.subheader(f"Select {right_name} Rows")
        # Show the dataframe with row indices visible
        st.dataframe(right_df, use_container_width=True)
        
        # Multi-select for row indices
        max_right_idx = len(right_df) - 1
        selected_right = st.multiselect(
            f"Select row indices from {right_name} (0-{max_right_idx})",
            options=list(range(len(right_df))),
            format_func=lambda x: f"Row {x}: {right_df.loc[x, right_key_cols[0]] if len(right_key_cols) > 0 else f'Index {x}'}"
        )
        
        # Display selected rows
        if selected_right:
            st.write("Selected rows:")
            st.dataframe(right_df.loc[selected_right], use_container_width=True)
    
    # --- Manual reconciliation with totals check ---
            if not selected_left or not selected_right:
                st.warning("Please select at least one row from each dataset")
            else:
                try:
                    # Initialize reconciliation engine
                    engine = DataReconciliationEngine()

                    # Prepare engine state
                    _ = engine.reconcile_datasets(
                        left_df=left_df.copy(),
                        right_df=right_df.copy(),
                        left_key_columns=left_key_cols,
                        right_key_columns=right_key_cols,
                        left_value_columns=left_value_cols,
                        right_value_columns=right_value_cols,
                        combine_duplicates=combine_duplicates,
                        partial_match_threshold=partial_threshold,
                        left_name=left_name,
                        right_name=right_name
                    )

                    # --- Check totals before enabling reconcile ---
                    validation = engine.validate_manual_matches([(l, r) for l in selected_left for r in selected_right])
                    totals_match = validation['totals_match']

                    if not totals_match:
                        st.warning("⚠ Totals from Left and Right do not match. Adjust your selection.")
                        st.json(validation['total_differences'])
                    else:
                        # Only show reconcile button if totals match
                        if st.button("🔗 Perform Manual Reconciliation", key="manual_reconcile_totals_match"):
                            with st.spinner("Performing manual reconciliation..."):
                                manual_results = engine.manual_reconcile(
                                    left_indexes=selected_left,
                                    right_indexes=selected_right,
                                    match_type='manual'
                                )

                                # Remove matched rows from unmatched list in UI
                                updated_unmatched = manual_results['unmatched_entries'].copy()
                                updated_unmatched = updated_unmatched[
                                    ~updated_unmatched['Matched'].astype(str).str.strip().astype(bool)
                                ]
                                st.session_state.unmatched_entries = updated_unmatched

                                # Save results for later use
                                st.session_state.reconciliation_results = manual_results
                                st.session_state.engine_initialized = True

                                st.success("Manual reconciliation completed successfully!")

                                # Show updated rows with match information
                                st.subheader("✅ Manual Reconciliation Results")
                                # You can expand this to show updated results here

                except Exception as e:
                    st.error(f"Error during manual reconciliation: {str(e)}")
                    st.exception(e)

                    
                    updated_col1, updated_col2 = st.columns(2)
                    
                    with updated_col1:
                        st.write(f"**{left_name} - Updated Rows:**")
                        left_display_cols = left_key_cols + left_value_cols + ['Matched', 'Match_Type', 'Match_Score']
                        left_updated_rows = manual_results['left_df_updated'].loc[selected_left, left_display_cols]
                        st.dataframe(left_updated_rows, use_container_width=True)
                    
                    with updated_col2:
                        st.write(f"**{right_name} - Updated Rows:**")
                        right_display_cols = right_key_cols + right_value_cols + ['Matched', 'Match_Type', 'Match_Score']
                        right_updated_rows = manual_results['right_df_updated'].loc[selected_right, right_display_cols]
                        st.dataframe(right_updated_rows, use_container_width=True)
                    
                    # Show match summary
                    st.subheader("📊 Manual Match Summary")
                    match_info_col1, match_info_col2 = st.columns(2)
                    
                    with match_info_col1:
                        st.metric("Left Rows Matched", len(selected_left))
                        left_totals = {}
                        for col in left_value_cols:
                            if pd.api.types.is_numeric_dtype(left_df[col]):
                                total = left_df.loc[selected_left, col].sum()
                                left_totals[col] = total
                                st.metric(f"Left Total - {col}", f"{total:,.2f}")
                    
                    with match_info_col2:
                        st.metric("Right Rows Matched", len(selected_right))
                        right_totals = {}
                        for i, col in enumerate(right_value_cols):
                            if pd.api.types.is_numeric_dtype(right_df[col]):
                                total = right_df.loc[selected_right, col].sum()
                                right_totals[col] = total
                                st.metric(f"Right Total - {col}", f"{total:,.2f}")
                                
                                # Show difference if corresponding left column exists
                                if i < len(left_value_cols):
                                    left_col = left_value_cols[i]
                                    if left_col in left_totals:
                                        diff = left_totals[left_col] - total
                                        st.metric(f"Difference ({left_col} - {col})", f"{diff:,.2f}")
                    
                except Exception as e:
                    st.error(f"Error during manual reconciliation: {str(e)}")
                    st.write("Debug information:")
                    st.write(f"Selected left indexes: {selected_left}")
                    st.write(f"Selected right indexes: {selected_right}")
                    st.write(f"Left DF shape: {left_df.shape}")
                    st.write(f"Right DF shape: {right_df.shape}")
                    st.exception(e)
    
    # Run Full Reconciliation Button (incorporating any manual matches)
    if st.button("🚀 Run Complete Reconciliation", type="primary", use_container_width=True):
        with st.spinner("Running complete reconciliation..."):
            try:
                # Use existing results if manual reconciliation was performed
                if st.session_state.reconciliation_results is not None:
                    st.info("Using existing manual matches and running complete reconciliation...")
                    results = st.session_state.reconciliation_results
                else:
                    # Run fresh reconciliation
                    engine = DataReconciliationEngine()
                    results = engine.reconcile_datasets(
                        left_df=left_df,
                        right_df=right_df,
                        left_key_columns=left_key_cols,
                        right_key_columns=right_key_cols,
                        left_value_columns=left_value_cols,
                        right_value_columns=right_value_cols,
                        combine_duplicates=combine_duplicates,
                        partial_match_threshold=partial_threshold,
                        left_name=left_name,
                        right_name=right_name
                    )
                
                st.success("Complete reconciliation finished!")
                
                # Display summary statistics
                st.header("📊 Reconciliation Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    left_key = f'total_{left_name.lower().replace(" ", "_")}_records'
                    st.metric(f"Total {left_name} Records", results['summary_stats'].get(left_key, 'N/A'))
                with col2:
                    right_key = f'total_{right_name.lower().replace(" ", "_")}_records'
                    st.metric(f"Total {right_name} Records", results['summary_stats'].get(right_key, 'N/A'))
                with col3:
                    st.metric("Matched Records", results['summary_stats'].get('matching_records', 'N/A'))
                with col4:
                    rate = results['summary_stats'].get('reconciliation_rate', 'N/A')
                    if isinstance(rate, (int, float)):
                        rate = f"{rate:.1f}%"
                    st.metric("Reconciliation Rate", rate)
                
                # Display final summary DataFrame
                st.header("📋 Final Summary DataFrame")
                st.markdown("*This shows the comprehensive reconciliation results by key*")
                final_summary = results['final_summary_df']
                st.dataframe(final_summary, use_container_width=True)
                
                # Download final summary
                st.header("💾 Download Results")
                csv_summary = final_summary.to_csv(index=False)
                st.download_button(
                    label="📥 Download Final Summary (CSV)",
                    data=csv_summary,
                    file_name="reconciliation_final_summary.csv",
                    mime="text/csv"
                )
                
                # Display reconciliation results in order
                st.header("🔍 Reconciliation Results")
                
                # 1. Matched entries
                st.subheader("✅ Matched Entries")
                if not results['matching_entries'].empty:
                    # Combine matched entries from both sides and remove duplicates for cleaner display
                    matched_entries = results['matching_entries'].copy()
                    st.dataframe(matched_entries, use_container_width=True)
                    st.write(f"**Count:** {len(matched_entries)} entries")
                else:
                    st.info("No matched entries found")
                
                # 2. Unmatched entries (combination of mismatched and partial matches)
                st.subheader("⚠️ Unmatched Entries")
                unmatched_list = []
                if not results['mismatched_entries'].empty:
                    unmatched_list.append(results['mismatched_entries'])
                if not results['partial_matches'].empty:
                    unmatched_list.append(results['partial_matches'])
                
                if 'unmatched_entries' in st.session_state:
                    unmatched_entries = st.session_state.unmatched_entries
                elif unmatched_list:
                    unmatched_entries = pd.concat(unmatched_list, ignore_index=True)
                    st.dataframe(unmatched_entries, use_container_width=True)
                    st.write(f"**Count:** {len(unmatched_entries)} entries")
                else:
                    st.info("No unmatched entries found")
                
                # 3. Left dataset only entries
                st.subheader(f"⬅️ {left_name} Only Entries")
                if not results['left_only_entries'].empty:
                    st.dataframe(results['left_only_entries'], use_container_width=True)
                    st.write(f"**Count:** {len(results['left_only_entries'])} entries")
                else:
                    st.info(f"No {left_name}-only entries found")
                
                # 4. Right dataset only entries
                st.subheader(f"➡️ {right_name} Only Entries")
                if not results['right_only_entries'].empty:
                    st.dataframe(results['right_only_entries'], use_container_width=True)
                    st.write(f"**Count:** {len(results['right_only_entries'])} entries")
                else:
                    st.info(f"No {right_name}-only entries found")
                
                # Display updated datasets with matches (moved below the main results)
                st.header("🔗 Updated Datasets with Matches")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"{left_name} with Matches")
                    left_updated = results['left_df_updated']
                    display_cols = left_key_cols + left_value_cols + ['Matched', 'Match_Type', 'Match_Score']
                    st.dataframe(left_updated[display_cols], use_container_width=True)
                
                with col2:
                    st.subheader(f"{right_name} with Matches")
                    right_updated = results['right_df_updated']
                    display_cols = right_key_cols + right_value_cols + ['Matched', 'Match_Type', 'Match_Score']
                    st.dataframe(right_updated[display_cols], use_container_width=True)
                
                if show_detailed_results:
                    # Detailed breakdown (kept as tabs for additional detail)
                    st.header("📊 Additional Details")
                    
                    # Create tabs for different categories
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Exact Matches", "Partial Matches", "Mismatched", "Left Only Details", "Right Only Details"
                    ])
                    
                    with tab1:
                        st.subheader("Exactly Matched Entries (Details)")
                        if not results['matching_entries'].empty:
                            # Filter for exact matches only
                            exact_matches = results['matching_entries'][
                                (results['matching_entries']['Match_Type'] == 'exact') | 
                                (results['matching_entries']['Match_Type'] == 'manual')
                            ]
                            if not exact_matches.empty:
                                st.dataframe(exact_matches, use_container_width=True)
                                st.write(f"**Count:** {len(exact_matches)} entries")
                            else:
                                st.info("No exact matches found")
                        else:
                            st.info("No exactly matched entries found")
                    
                    with tab2:
                        st.subheader("Partial Matches (Details)")
                        if not results['partial_matches'].empty:
                            st.dataframe(results['partial_matches'], use_container_width=True)
                            st.write(f"**Count:** {len(results['partial_matches'])} entries")
                        else:
                            st.info("No partial matches found")
                    
                    with tab3:
                        st.subheader("Mismatched Entries (Details)")
                        if not results['mismatched_entries'].empty:
                            st.dataframe(results['mismatched_entries'], use_container_width=True)
                            st.write(f"**Count:** {len(results['mismatched_entries'])} entries")
                        else:
                            st.info("No mismatched entries found")
                    
                    with tab4:
                        st.subheader(f"{left_name} Only Entries (Details)")
                        if not results['left_only_entries'].empty:
                            st.dataframe(results['left_only_entries'], use_container_width=True)
                            st.write(f"**Count:** {len(results['left_only_entries'])} entries")
                        else:
                            st.info(f"No {left_name}-only entries found")
                    
                    with tab5:
                        st.subheader(f"{right_name} Only Entries (Details)")
                        if not results['right_only_entries'].empty:
                            st.dataframe(results['right_only_entries'], use_container_width=True)
                            st.write(f"**Count:** {len(results['right_only_entries'])} entries")
                        else:
                            st.info(f"No {right_name}-only entries found")
                
                # Reconciliation statement
                st.header("📄 Reconciliation Statement")
                st.text(results['reconciliation_statement'])
                
                # Download reconciliation statement
                st.download_button(
                    label="📥 Download Reconciliation Statement",
                    data=results['reconciliation_statement'],
                    file_name="reconciliation_statement.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"An error occurred during reconciliation: {str(e)}")
                st.exception(e)

# Clear Session State Button
if st.button("🔄 Reset All Data", help="Clear all session state and start fresh"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Session state cleared! Please refresh the page to start fresh.")
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
### 💡 Tips for Manual Reconciliation:
1. **Row Selection**: Use the row indices (0, 1, 2, etc.) to select specific rows
2. **View Data**: The dataframes show all data with visible row indices
3. **Multiple Rows**: You can select multiple rows from each dataset to match together  
4. **Manual First**: Perform manual reconciliation first, then run complete reconciliation
5. **Session State**: Manual matches are preserved until you reset or refresh

### 🔧 Key Features:
- ✅ Manual row selection with clear row indices
- ✅ Preview of selected rows before matching
- ✅ Match summary with totals and differences
- ✅ Preserved manual matches in complete reconciliation
- ✅ Session state management for workflow continuity
""")