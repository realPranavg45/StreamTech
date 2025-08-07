"""
Column Mapping Module
====================

Handles column mapping operations for datasets, including validation of mappings
and transformation of DataFrames according to mapping configurations.

Author: Assistant
Version: 1.0
"""

import pandas as pd
from typing import Dict, List, Set, Optional, Tuple


class ColumnMappingError(Exception):
    """Custom exception for column mapping errors."""
    pass


class ColumnMapper:
    """Class to handle column mapping operations."""
    
    def __init__(self, column_mapping: Dict[str, str]):
        """
        Initialize the ColumnMapper.
        
        Parameters:
        -----------
        column_mapping : Dict[str, str]
            Mapping from old column names to new/standard column names
            Format: {'old_column_name': 'new_column_name'}
        """
        self.column_mapping = column_mapping
        self.reverse_mapping = {v: k for k, v in column_mapping.items()}
    
    def validate_mapping(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that the column mapping can be applied to the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to validate against
            
        Returns:
        --------
        Tuple[bool, List[str]]
            (is_valid, list_of_missing_columns)
        """
        df_columns = set(df.columns)
        mapping_columns = set(self.column_mapping.keys())
        missing_columns = mapping_columns - df_columns
        
        return len(missing_columns) == 0, list(missing_columns)
    
    def apply_mapping(self, df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
        """
        Apply column mapping to the DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to transform
        strict : bool, default True
            If True, raises error if any mapped columns are missing
            If False, only maps available columns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with renamed columns
            
        Raises:
        -------
        ColumnMappingError
            If strict=True and mapped columns are missing
        """
        if strict:
            is_valid, missing_columns = self.validate_mapping(df)
            if not is_valid:
                raise ColumnMappingError(f"Cannot apply mapping. Missing columns: {missing_columns}")
        
        # Only apply mapping for columns that exist
        available_mapping = {k: v for k, v in self.column_mapping.items() if k in df.columns}
        
        try:
            df_mapped = df.rename(columns=available_mapping)
            return df_mapped
        except Exception as e:
            raise ColumnMappingError(f"Error applying column mapping: {str(e)}")
    
    def get_mapped_column_name(self, original_name: str) -> str:
        """
        Get the mapped column name for an original column name.
        
        Parameters:
        -----------
        original_name : str
            Original column name
            
        Returns:
        --------
        str
            Mapped column name, or original name if no mapping exists
        """
        return self.column_mapping.get(original_name, original_name)
    
    def get_original_column_name(self, mapped_name: str) -> str:
        """
        Get the original column name for a mapped column name.
        
        Parameters:
        -----------
        mapped_name : str
            Mapped column name
            
        Returns:
        --------
        str
            Original column name, or mapped name if no reverse mapping exists
        """
        return self.reverse_mapping.get(mapped_name, mapped_name)
    
    def get_required_columns_for_validation(self, validation_config: Dict) -> Set[str]:
        """
        Extract all column names required by validation configuration.
        
        Parameters:
        -----------
        validation_config : Dict
            Validation configuration dictionary
            
        Returns:
        --------
        Set[str]
            Set of all column names referenced in validation config
        """
        required_columns = set()
        
        for config_key, config_value in validation_config.items():
            if isinstance(config_value, list):
                if config_key == 'composite_unique':
                    # Handle nested list structure for composite unique constraints
                    for combo in config_value:
                        required_columns.update(combo)
                else:
                    required_columns.update(config_value)
            elif isinstance(config_value, dict):
                required_columns.update(config_value.keys())
        
        return required_columns
    
    def validate_columns_exist_after_mapping(self, df: pd.DataFrame, validation_config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate that all required columns for validation exist after mapping.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The original DataFrame
        validation_config : Dict
            Validation configuration dictionary
            
        Returns:
        --------
        Tuple[bool, List[str]]
            (all_exist, list_of_missing_columns_after_mapping)
        """
        # Apply mapping first
        df_mapped = self.apply_mapping(df, strict=False)
        
        # Get required columns from validation config
        required_columns = self.get_required_columns_for_validation(validation_config)
        
        # Check which columns are missing after mapping
        available_columns = set(df_mapped.columns)
        missing_columns = required_columns - available_columns
        
        return len(missing_columns) == 0, list(missing_columns)
    
    def get_mapping_summary(self) -> Dict:
        """
        Get a summary of the column mapping configuration.
        
        Returns:
        --------
        Dict
            Summary information about the mapping
        """
        return {
            'total_mappings': len(self.column_mapping),
            'mappings': self.column_mapping,
            'reverse_mappings': self.reverse_mapping
        }


def create_identity_mapping(columns: List[str]) -> Dict[str, str]:
    """
    Create an identity mapping where each column maps to itself.
    
    Parameters:
    -----------
    columns : List[str]
        List of column names
        
    Returns:
    --------
    Dict[str, str]
        Identity mapping dictionary
    """
    return {col: col for col in columns}


def merge_mappings(*mappings: Dict[str, str]) -> Dict[str, str]:
    """
    Merge multiple column mappings into one.
    Later mappings override earlier ones for duplicate keys.
    
    Parameters:
    -----------
    *mappings : Dict[str, str]
        Variable number of mapping dictionaries
        
    Returns:
    --------
    Dict[str, str]
        Merged mapping dictionary
    """
    merged = {}
    for mapping in mappings:
        merged.update(mapping)
    return merged


def invert_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Invert a column mapping dictionary.
    
    Parameters:
    -----------
    mapping : Dict[str, str]
        Original mapping dictionary
        
    Returns:
    --------
    Dict[str, str]
        Inverted mapping dictionary
    """
    return {v: k for k, v in mapping.items()}