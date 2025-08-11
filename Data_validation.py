"""
Data Validation Module
======================

Comprehensive data validation functionality for datasets.
Validates datasets based on various constraints like uniqueness, null checks, 
data types, ranges, and categorical values.

Author: Assistant
Version: 1.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import re


class DatasetValidationResult:
    """Class to store validation results with detailed error information."""
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.summary = {}
    
    def add_error(self, error_type: str, column: str, details: str, invalid_rows: List[int] = None):
        """Add validation error."""
        self.is_valid = False
        error_info = {
            'error_type': error_type,
            'column': column,
            'details': details,
            'invalid_rows': invalid_rows or []
        }
        self.errors.append(error_info)
    
    def add_warning(self, warning_type: str, column: str, details: str):
        """Add validation warning."""
        warning_info = {
            'warning_type': warning_type,
            'column': column,
            'details': details
        }
        self.warnings.append(warning_info)
    
    def get_summary(self):
        """Get validation summary."""
        return {
            'is_valid': self.is_valid,
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def merge(self, other_result: 'DatasetValidationResult'):
        """
        Merge another validation result into this one.
        
        Parameters:
        -----------
        other_result : DatasetValidationResult
            Another validation result to merge
        """
        self.errors.extend(other_result.errors)
        self.warnings.extend(other_result.warnings)
        if not other_result.is_valid:
            self.is_valid = False


class DataValidator:
    """Main class for performing data validation operations."""
    
    def __init__(self, validation_config: Dict[str, Any]):
        """
        Initialize the DataValidator.
        
        Parameters:
        -----------
        validation_config : Dict[str, Any]
            Validation configuration with the following structure:
            {
                'unique_columns': List[str],  # Columns that must have unique values
                'composite_unique': List[List[str]],  # Combinations that must be collectively unique
                'non_null_columns': List[str],  # Columns that cannot be null/blank
                'numeric_columns': Dict[str, Dict],  # Numeric validation with min/max
                'date_columns': Dict[str, Dict],  # Date validation with format and range
                'categorical_columns': Dict[str, List],  # Columns with allowed values
            }
        """
        self.validation_config = validation_config
        self.result = DatasetValidationResult()
    
    def validate(self, df: pd.DataFrame) -> DatasetValidationResult:
        """
        Perform comprehensive validation on the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to validate
            
        Returns:
        --------
        DatasetValidationResult
            Validation results with errors and warnings
        """
        self.result = DatasetValidationResult()
        
        # Perform all validation checks
        self._validate_unique_columns(df)
        self._validate_composite_unique(df)
        self._validate_non_null_columns(df)
        self._validate_numeric_columns(df)
        self._validate_date_columns(df)
        self._validate_categorical_columns(df)
        
        # Generate summary statistics
        self.result.summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'validation_rules_applied': len([k for k in self.validation_config.keys() if self.validation_config[k]])
        }
        
        return self.result
    
    def _validate_unique_columns(self, df: pd.DataFrame):
        """Validate unique column constraints."""
        if 'unique_columns' not in self.validation_config:
            return
            
        for column in self.validation_config['unique_columns']:
            if column in df.columns:
                duplicates = df[df.duplicated(subset=[column], keep=False)]
                if not duplicates.empty:
                    duplicate_rows = duplicates.index.tolist()
                    duplicate_values = duplicates[column].unique().tolist()
                    self.result.add_error(
                        'unique_constraint', column,
                        f"Found {len(duplicate_rows)} duplicate rows with values: {duplicate_values}",
                        duplicate_rows
                    )
    
    def _validate_composite_unique(self, df: pd.DataFrame):
        """Validate composite unique constraints."""
        if 'composite_unique' not in self.validation_config:
            return
        
        for combo in self.validation_config['composite_unique']:
            # Ensure combo is a list/tuple
            if not isinstance(combo, (list, tuple)):
                self.result.add_warning(
                    'config_warning', str(combo),
                    f"Invalid composite unique definition: {combo} (must be list of column names)"
                )
                continue

            # Keep only columns that exist in df
            valid_cols = [col for col in combo if col in df.columns]
            missing_cols = [col for col in combo if col not in df.columns]

            # Warn if any columns are missing
            if missing_cols:
                self.result.add_warning(
                    'missing_column', '+'.join(combo),
                    f"Missing columns in DataFrame: {missing_cols}"
                )
            
            # Skip if fewer than 2 valid columns remain
            if len(valid_cols) < len(combo):
                continue

            # Check duplicates
            duplicates = df[df.duplicated(subset=valid_cols, keep=False)]
            if not duplicates.empty:
                duplicate_rows = duplicates.index.tolist()
                self.result.add_error(
                    'composite_unique_constraint', '+'.join(valid_cols),
                    f"Found {len(duplicate_rows)} rows with duplicate combinations",
                    duplicate_rows
                )

        
    def _validate_non_null_columns(self, df: pd.DataFrame):
        """Validate non-null column constraints."""
        if 'non_null_columns' not in self.validation_config:
            return
            
        for column in self.validation_config['non_null_columns']:
            if column in df.columns:
                null_mask = df[column].isna() | (df[column].astype(str).str.strip() == '')
                null_rows = df[null_mask].index.tolist()
                if null_rows:
                    self.result.add_error(
                        'null_constraint', column,
                        f"Found {len(null_rows)} null/blank values",
                        null_rows
                    )
    
    def _validate_numeric_columns(self, df: pd.DataFrame):
        """Validate numeric column constraints."""
        if 'numeric_columns' not in self.validation_config:
            return
            
        for column, constraints in self.validation_config['numeric_columns'].items():
            if column in df.columns:
                # Check if values are numeric
                non_numeric_mask = pd.to_numeric(df[column], errors='coerce').isna()
                non_numeric_rows = df[non_numeric_mask & df[column].notna()].index.tolist()
                
                if non_numeric_rows:
                    self.result.add_error(
                        'numeric_constraint', column,
                        f"Found {len(non_numeric_rows)} non-numeric values",
                        non_numeric_rows
                    )
                else:
                    # Check min/max constraints
                    numeric_series = pd.to_numeric(df[column], errors='coerce')
                    
                    if 'min' in constraints:
                        min_violation_mask = numeric_series < constraints['min']
                        min_violation_rows = df[min_violation_mask].index.tolist()
                        if min_violation_rows:
                            self.result.add_error(
                                'numeric_min_constraint', column,
                                f"Found {len(min_violation_rows)} values below minimum {constraints['min']}",
                                min_violation_rows
                            )
                    
                    if 'max' in constraints:
                        max_violation_mask = numeric_series > constraints['max']
                        max_violation_rows = df[max_violation_mask].index.tolist()
                        if max_violation_rows:
                            self.result.add_error(
                                'numeric_max_constraint', column,
                                f"Found {len(max_violation_rows)} values above maximum {constraints['max']}",
                                max_violation_rows
                            )
    
    def _validate_date_columns(self, df: pd.DataFrame):
        """Validate date column constraints."""
        if 'date_columns' not in self.validation_config:
            return
            
        for column, constraints in self.validation_config['date_columns'].items():
            if column in df.columns:
                date_format = constraints.get('format', 'dd/mm/yyyy')
                
                # Convert format to pandas format
                pandas_format = self._convert_date_format(date_format)
                
                # Try to parse dates
                invalid_dates = []
                valid_dates = []
                
                for idx, date_str in df[column].items():
                    if pd.isna(date_str) or str(date_str).strip() == '':
                        continue
                    
                    try:
                        parsed_date = datetime.strptime(str(date_str), pandas_format)
                        valid_dates.append((idx, parsed_date))
                    except (ValueError, TypeError):
                        invalid_dates.append(idx)
                
                if invalid_dates:
                    self.result.add_error(
                        'date_format_constraint', column,
                        f"Found {len(invalid_dates)} invalid date formats (expected: {date_format})",
                        invalid_dates
                    )
                
                # Check date range constraints
                self._validate_date_ranges(column, constraints, valid_dates, pandas_format)
    
    def _validate_date_ranges(self, column: str, constraints: Dict, valid_dates: List, pandas_format: str):
        """Validate date range constraints."""
        if not valid_dates or ('min_date' not in constraints and 'max_date' not in constraints):
            return
            
        if 'min_date' in constraints:
            try:
                min_date = datetime.strptime(constraints['min_date'], pandas_format)
                early_dates = [idx for idx, date in valid_dates if date < min_date]
                if early_dates:
                    self.result.add_error(
                        'date_min_constraint', column,
                        f"Found {len(early_dates)} dates before {constraints['min_date']}",
                        early_dates
                    )
            except ValueError:
                self.result.add_warning(
                    'config_warning', column,
                    f"Invalid min_date format in config: {constraints['min_date']}"
                )
        
        if 'max_date' in constraints:
            try:
                max_date = datetime.strptime(constraints['max_date'], pandas_format)
                late_dates = [idx for idx, date in valid_dates if date > max_date]
                if late_dates:
                    self.result.add_error(
                        'date_max_constraint', column,
                        f"Found {len(late_dates)} dates after {constraints['max_date']}",
                        late_dates
                    )
            except ValueError:
                self.result.add_warning(
                    'config_warning', column,
                    f"Invalid max_date format in config: {constraints['max_date']}"
                )
    
    def _validate_categorical_columns(self, df: pd.DataFrame):
        """Validate categorical column constraints."""
        if 'categorical_columns' not in self.validation_config:
            return
            
        for column, allowed_values in self.validation_config['categorical_columns'].items():
            if column in df.columns:
                # Remove null values for validation
                non_null_mask = df[column].notna() & (df[column].astype(str).str.strip() != '')
                non_null_values = df[non_null_mask][column]
                
                invalid_mask = ~non_null_values.isin(allowed_values)
                invalid_rows = non_null_values[invalid_mask].index.tolist()
                
                if invalid_rows:
                    invalid_values = non_null_values[invalid_mask].unique().tolist()
                    self.result.add_error(
                        'categorical_constraint', column,
                        f"Found {len(invalid_rows)} invalid values: {invalid_values}. Allowed: {allowed_values}",
                        invalid_rows
                    )
    
    def _convert_date_format(self, date_format: str) -> str:
        """
        Convert custom date format to pandas strptime format.
        
        Parameters:
        -----------
        date_format : str
            Custom date format string
            
        Returns:
        --------
        str
            Pandas strptime format string
        """
        format_mapping = {
            'dd/mm/yyyy': '%d/%m/%Y',
            'mm/dd/yyyy': '%m/%d/%Y',
            'yyyy-mm-dd': '%Y-%m-%d',
            'dd-mm-yyyy': '%d-%m-%Y',
            'mm-dd-yyyy': '%m-%d-%Y',
            'dd.mm.yyyy': '%d.%m.%Y',
            'yyyy/mm/dd': '%Y/%m/%d'
        }
        
        return format_mapping.get(date_format.lower(), '%d/%m/%Y')


def validate_single_column(df: pd.DataFrame, column: str, validation_type: str, 
                          constraints: Any = None) -> DatasetValidationResult:
    """
    Validate a single column with specific constraints.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the column
    column : str
        Column name to validate
    validation_type : str
        Type of validation ('unique', 'non_null', 'numeric', 'date', 'categorical')
    constraints : Any
        Validation constraints specific to the validation type
        
    Returns:
    --------
    DatasetValidationResult
        Validation results for the single column
    """
    config = {validation_type + '_columns': {column: constraints} if constraints else [column]}
    validator = DataValidator(config)
    return validator.validate(df)


def quick_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a quick data profile for basic validation insights.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to profile
        
    Returns:
    --------
    Dict[str, Any]
        Data profile summary
    """
    profile = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': {},
        'missing_data_summary': {},
        'potential_issues': []
    }
    
    for column in df.columns:
        col_info = {
            'dtype': str(df[column].dtype),
            'null_count': df[column].isna().sum(),
            'null_percentage': (df[column].isna().sum() / len(df)) * 100,
            'unique_count': df[column].nunique(),
            'duplicate_count': df.duplicated(subset=[column]).sum()
        }
        
        # Add type-specific info
        if pd.api.types.is_numeric_dtype(df[column]):
            col_info.update({
                'min_value': df[column].min(),
                'max_value': df[column].max(),
                'mean_value': df[column].mean()
            })
        
        profile['columns'][column] = col_info
        
        # Identify potential issues
        if col_info['null_percentage'] > 50:
            profile['potential_issues'].append(f"Column '{column}' has >50% missing values")
        
        if col_info['duplicate_count'] > 0 and col_info['unique_count'] < len(df) * 0.9:
            profile['potential_issues'].append(f"Column '{column}' has many duplicate values")
    
    return profile
