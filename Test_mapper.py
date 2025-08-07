"""
Streamlit Testing Script for Column Mapping Module
==================================================

Interactive testing interface for column mapping between old and new datasets.
Tests the backend logic that maps columns between different dataset versions.

Run with: streamlit run column_mapper_test.py

Author: Assistant  
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import Dict, List
import io

# Import the column mapping module (assuming it's in the same directory)
try:
    from Column_mapper import (
        ColumnMapper, ColumnMappingError, create_identity_mapping, 
        merge_mappings, invert_mapping
    )
except ImportError:
    st.error("Could not import column_mapper module. Please ensure column_mapper.py is in the same directory.")
    st.stop()


def create_old_dataset() -> pd.DataFrame:
    """Create sample OLD dataset with legacy column names."""
    np.random.seed(42)
    data = {
        'emp_id': range(1, 51),
        'fname': [f'John{i}' for i in range(1, 51)],
        'lname': [f'Smith{i}' for i in range(1, 51)],
        'dept_name': np.random.choice(['HR', 'IT', 'Finance', 'Marketing'], 50),
        'salary_amount': np.random.randint(30000, 100000, 50),
        'joining_dt': pd.date_range('2020-01-01', periods=50, freq='D'),
        'emp_status': np.random.choice(['Active', 'Inactive'], 50),
        'mgr_id': np.random.randint(1, 11, 50),
        'contact_no': [f'+1-555-{1000+i:04d}' for i in range(50)],
        'email_addr': [f'emp{i}@oldcompany.com' for i in range(1, 51)]
    }
    return pd.DataFrame(data)


def create_new_dataset() -> pd.DataFrame:
    """Create sample NEW dataset with updated column names."""
    np.random.seed(123)
    data = {
        'employee_id': range(101, 151),
        'first_name': [f'Jane{i}' for i in range(1, 51)],
        'last_name': [f'Doe{i}' for i in range(1, 51)],
        'department': np.random.choice(['Human Resources', 'Information Technology', 'Finance', 'Marketing'], 50),
        'salary': np.random.randint(35000, 120000, 50),
        'hire_date': pd.date_range('2023-01-01', periods=50, freq='D'),
        'status': np.random.choice(['Active', 'Terminated', 'On Leave'], 50),
        'manager_id': np.random.randint(1, 11, 50),
        'phone_number': [f'+1-555-{2000+i:04d}' for i in range(50)],
        'email': [f'employee{i}@newcompany.com' for i in range(101, 151)]
    }
    return pd.DataFrame(data)


def get_standard_mapping() -> Dict[str, str]:
    """Get standard mapping from old column names to new column names."""
    return {
        'emp_id': 'employee_id',
        'fname': 'first_name', 
        'lname': 'last_name',
        'dept_name': 'department',
        'salary_amount': 'salary',
        'joining_dt': 'hire_date',
        'emp_status': 'status',
        'mgr_id': 'manager_id',
        'contact_no': 'phone_number',
        'email_addr': 'email'
    }


def create_validation_config() -> Dict:
    """Create validation config that works with standardized column names."""
    return {
        'unique_columns': ['employee_id', 'email'],
        'composite_unique': [['first_name', 'last_name', 'department']],
        'non_null_columns': ['employee_id', 'first_name', 'last_name', 'department'],
        'numeric_columns': {
            'salary': {'min': 25000, 'max': 150000},
            'manager_id': {'min': 1, 'max': 20}
        },
        'categorical_columns': {
            'department': ['Human Resources', 'Information Technology', 'Finance', 'Marketing', 'HR', 'IT'],
            'status': ['Active', 'Inactive', 'Terminated', 'On Leave']
        }
    }


def main():
    st.set_page_config(
        page_title="Column Mapping Backend Tester",
        page_icon="🔄",
        layout="wide"
    )
    
    st.title("🔄 Column Mapping Backend Tester")
    st.markdown("Test column mapping between **Old Dataset** and **New Dataset** formats")
    
    # Sidebar for dataset selection
    st.sidebar.header("📊 Dataset Configuration")
    
    # Old dataset configuration
    st.sidebar.subheader("Old Dataset")
    old_data_source = st.sidebar.selectbox(
        "Select Old Dataset Source",
        ["Upload CSV", "Upload Excel"],
        key="old_source"
    )
    
    old_df = None
    if old_data_source == "Sample Old Data":
        old_df = create_old_dataset()
        st.sidebar.success(f"✅ Old dataset: {old_df.shape[0]} rows, {old_df.shape[1]} cols")
    elif old_data_source == "Upload CSV":
        old_file = st.sidebar.file_uploader("Upload Old Dataset CSV", type=['csv'], key="old_csv")
        if old_file:
            old_df = pd.read_csv(old_file)
            st.sidebar.success(f"✅ Old CSV: {old_df.shape[0]} rows, {old_df.shape[1]} cols")
    elif old_data_source == "Upload Excel":
        old_file = st.sidebar.file_uploader("Upload Old Dataset Excel", type=['xlsx', 'xls'], key="old_excel")
        if old_file:
            old_df = pd.read_excel(old_file)
            st.sidebar.success(f"✅ Old Excel: {old_df.shape[0]} rows, {old_df.shape[1]} cols")
    
    # New dataset configuration
    st.sidebar.subheader("New Dataset")
    new_data_source = st.sidebar.selectbox(
        "Select New Dataset Source", 
        [ "Upload CSV", "Upload Excel"],
        key="new_source"
    )
    
    new_df = None
    if new_data_source == "Sample New Data":
        new_df = create_new_dataset()
        st.sidebar.success(f"✅ New dataset: {new_df.shape[0]} rows, {new_df.shape[1]} cols")
    elif new_data_source == "Upload CSV":
        new_file = st.sidebar.file_uploader("Upload New Dataset CSV", type=['csv'], key="new_csv")
        if new_file:
            new_df = pd.read_csv(new_file)
            st.sidebar.success(f"✅ New CSV: {new_df.shape[0]} rows, {new_df.shape[1]} cols")
    elif new_data_source == "Upload Excel":
        new_file = st.sidebar.file_uploader("Upload New Dataset Excel", type=['xlsx', 'xls'], key="new_excel")
        if new_file:
            new_df = pd.read_excel(new_file)
            st.sidebar.success(f"✅ New Excel: {new_df.shape[0]} rows, {new_df.shape[1]} cols")
    
    if old_df is None or new_df is None:
        st.warning("⚠️ Please load both Old and New datasets to continue testing")
        return
    
    # Display datasets side by side
    st.header("📋 Dataset Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗂️ Old Dataset")
        st.dataframe(old_df.head(8), use_container_width=True)
        st.write("**Old Column Names:**")
        old_cols_text = ", ".join([f"`{col}`" for col in old_df.columns])
        st.markdown(old_cols_text)
    
    with col2:
        st.subheader("🆕 New Dataset") 
        st.dataframe(new_df.head(8), use_container_width=True)
        st.write("**New Column Names:**")
        new_cols_text = ", ".join([f"`{col}`" for col in new_df.columns])
        st.markdown(new_cols_text)
    
    # Column mapping configuration
    st.header("🗂️ Backend Column Mapping Configuration")
    
    mapping_method = st.radio(
        "Choose mapping configuration method:",
        [ "Manual Mapping Creation", "JSON Import"]
    )
    
    column_mapping = {}
    
    if mapping_method == "Auto-detect Standard Mapping":
        column_mapping = get_standard_mapping()
        st.success("✅ Using standard enterprise mapping configuration")
        
        # Show mapping preview
        with st.expander("📋 View Standard Mapping"):
            mapping_df = pd.DataFrame([
                {"Old Column": k, "New Column": v, "Status": "✅ Mapped"} 
                for k, v in column_mapping.items()
            ])
            st.dataframe(mapping_df, use_container_width=True)
    
    elif mapping_method == "Manual Mapping Creation":
        st.write("**Create Backend Mapping Rules:**")
        st.info("💡 Map old dataset columns to new dataset column names")
        
        # Initialize session state for mappings
        if 'manual_mappings' not in st.session_state:
            st.session_state.manual_mappings = []
        
        # Add mapping interface
        col1, col2, col3 = st.columns([3, 3, 1])
        
        with col1:
            old_col = st.selectbox("Old Dataset Column", [""] + list(old_df.columns))
        with col2:
            new_col = st.selectbox("Maps to New Dataset Column", [""] + list(new_df.columns))
        with col3:
            if st.button("➕ Add"):
                if old_col and new_col:
                    st.session_state.manual_mappings.append((old_col, new_col))
                    st.rerun()
        
        # Display current mappings
        if st.session_state.manual_mappings:
            st.write("**Current Mapping Rules:**")
            for i, (old, new) in enumerate(st.session_state.manual_mappings):
                col1, col2, col3 = st.columns([3, 3, 1])
                with col1:
                    st.text(old)
                with col2:
                    st.text(f"→ {new}")
                with col3:
                    if st.button("🗑️", key=f"del_{i}"):
                        st.session_state.manual_mappings.pop(i)
                        st.rerun()
                
                column_mapping[old] = new
        
        if st.button("🔄 Clear All Mappings"):
            st.session_state.manual_mappings = []
            st.rerun()

        
    
    elif mapping_method == "JSON Import":
        json_mapping = st.text_area(
            "Import mapping configuration as JSON:",
            value=json.dumps(get_standard_mapping(), indent=2),
            height=200
        )
        try:
            column_mapping = json.loads(json_mapping)
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON format")
            return
    
    if not column_mapping:
        st.warning("⚠️ Please configure column mapping rules to test backend functionality")
        return
    
    # Initialize backend column mapper
    try:
        backend_mapper = ColumnMapper(column_mapping)
        st.success("✅ Backend Column Mapper initialized successfully!")
    except Exception as e:
        st.error(f"❌ Backend initialization error: {e}")
        return
    
    # Backend testing tabs
    st.header("🧪 Backend Column Mapping Tests")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Mapping Validation", "Transform Old→New", "Reverse Transform", 
        "Validation Integration", "Backend Performance"
    ])
    
    with tab1:
        st.subheader("🔍 Backend Mapping Validation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Validate Against Old Dataset:**")
            old_valid, old_missing = backend_mapper.validate_mapping(old_df)
            
            if old_valid:
                st.success("✅ All mapped columns exist in old dataset")
            else:
                st.error(f"❌ Missing in old dataset: {old_missing}")
            
            # Check coverage
            old_coverage = len(column_mapping) / len(old_df.columns) * 100
            st.metric("Mapping Coverage", f"{old_coverage:.1f}%")
        
        with col2:
            st.write("**Check Against New Dataset:**")
            # Check if mapped names exist in new dataset
            mapped_names = list(column_mapping.values())
            new_cols = set(new_df.columns)
            missing_in_new = [name for name in mapped_names if name not in new_cols]
            
            if not missing_in_new:
                st.success("✅ All mapped names exist in new dataset")
            else:
                st.error(f"❌ Missing in new dataset: {missing_in_new}")
            
            # Check reverse coverage
            new_coverage = len([col for col in new_df.columns if col in mapped_names]) / len(new_df.columns) * 100
            st.metric("New Dataset Coverage", f"{new_coverage:.1f}%")
    
    with tab2:
        st.subheader("🔄 Transform Old Dataset → New Format")
        
        transform_mode = st.radio("Transformation Mode:", ["Strict", "Flexible"])
        
        try:
            # Apply transformation
            old_transformed = backend_mapper.apply_mapping(
                old_df, 
                strict=(transform_mode == "Strict")
            )
            
            st.success(f"✅ Successfully transformed old dataset!")
            
            # Show transformation results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Before Transformation:**")
                st.dataframe(old_df.head(5), use_container_width=True)
            
            with col2:
                st.write("**After Transformation:**") 
                st.dataframe(old_transformed.head(5), use_container_width=True)
            
            # Column mapping summary
            st.write("**Transformation Summary:**")
            transform_summary = []
            for old_col in old_df.columns:
                new_col = backend_mapper.get_mapped_column_name(old_col)
                status = "✅ Mapped" if old_col != new_col else "➡️ Unchanged"
                transform_summary.append({
                    "Original": old_col,
                    "Transformed": new_col, 
                    "Status": status
                })
            
            summary_df = pd.DataFrame(transform_summary)
            st.dataframe(summary_df, use_container_width=True)
            
            # Download transformed data
            csv_buffer = io.StringIO()
            old_transformed.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Transformed Dataset",
                data=csv_buffer.getvalue(),
                file_name="old_dataset_transformed.csv",
                mime="text/csv"
            )
            
        except ColumnMappingError as e:
            st.error(f"❌ Transformation failed: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
    
    with tab3:
        st.subheader("↩️ Reverse Transformation (New → Old)")
        
        st.info("💡 Testing reverse mapping to transform new dataset format back to old format")
        
        # Create reverse mapper
        reverse_mapping = invert_mapping(column_mapping)
        reverse_mapper = ColumnMapper(reverse_mapping)
        
        try:
            # Apply reverse transformation
            new_to_old = reverse_mapper.apply_mapping(new_df, strict=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**New Dataset (Original):**")
                st.dataframe(new_df.head(5), use_container_width=True)
            
            with col2:
                st.write("**New Dataset → Old Format:**")
                st.dataframe(new_to_old.head(5), use_container_width=True)
            
            st.success("✅ Reverse transformation completed!")
            
        except Exception as e:
            st.error(f"❌ Reverse transformation failed: {e}")
    
    with tab4:
        st.subheader("🔗 Backend Validation Integration")
        
        st.info("💡 Testing how column mapping integrates with validation rules")
        
        # Load validation config
        validation_config = create_validation_config()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Validation Config:**")
            st.json(validation_config)
        
        with col2:
            # Test required columns after mapping
            required_cols = backend_mapper.get_required_columns_for_validation(validation_config)
            st.write(f"**Required Columns for Validation:** {list(required_cols)}")
            
            # Test old dataset
            old_valid, old_missing = backend_mapper.validate_columns_exist_after_mapping(old_df, validation_config)
            if old_valid:
                st.success("✅ Old dataset → All validation columns available after mapping")
            else:
                st.error(f"❌ Old dataset → Missing after mapping: {old_missing}")
            
            # Test new dataset  
            new_valid, new_missing = backend_mapper.validate_columns_exist_after_mapping(new_df, validation_config)
            if new_valid:
                st.success("✅ New dataset → All validation columns available")
            else:
                st.warning(f"⚠️ New dataset → Missing: {new_missing}")
    
    with tab5:
        st.subheader("⚡ Backend Performance Testing")
        
        if st.button("🚀 Run Performance Tests"):
            import time
            
            # Create larger test datasets
            with st.spinner("Creating large test datasets..."):
                large_old = pd.concat([old_df] * 20, ignore_index=True)  # 20x larger
                large_new = pd.concat([new_df] * 20, ignore_index=True)  # 20x larger
            
            st.write(f"**Test Dataset Sizes:** {large_old.shape[0]:,} rows each")
            
            # Performance metrics
            results = {}
            
            # Test 1: Mapping validation speed
            start_time = time.time()
            is_valid, missing = backend_mapper.validate_mapping(large_old)
            results['validation_time'] = time.time() - start_time
            
            # Test 2: Transformation speed
            start_time = time.time()
            transformed = backend_mapper.apply_mapping(large_old, strict=False)
            results['transformation_time'] = time.time() - start_time
            
            # Test 3: Reverse mapping speed
            start_time = time.time()
            reverse_mapped = invert_mapping(column_mapping)
            results['reverse_mapping_time'] = time.time() - start_time
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Validation Time", f"{results['validation_time']:.4f}s")
            with col2:
                st.metric("Transformation Time", f"{results['transformation_time']:.4f}s") 
            with col3:
                st.metric("Reverse Mapping Time", f"{results['reverse_mapping_time']:.4f}s")
            
            # Memory usage estimation
            memory_usage = transformed.memory_usage(deep=True).sum() / (1024**2)  # MB
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
            
            st.success("✅ Performance testing completed!")
    
    # Footer with backend info
    st.markdown("---")
    st.markdown("**Backend Column Mapping Tester** - Testing enterprise data transformation logic")
    
    # Export configuration
    with st.expander("📤 Export Backend Configuration"):
        config_export = {
            "column_mapping": column_mapping,
            "validation_config": create_validation_config(),
            "old_dataset_columns": list(old_df.columns),
            "new_dataset_columns": list(new_df.columns)
        }
        
        st.code(json.dumps(config_export, indent=2), language="json")
        
        st.download_button(
            label="💾 Download Backend Config",
            data=json.dumps(config_export, indent=2),
            file_name="backend_mapping_config.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()