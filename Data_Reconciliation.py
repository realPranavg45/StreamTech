import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from copy import deepcopy
import warnings

class DataReconciliationEngine:
    """
    Universal data reconciliation engine that can match and reconcile any two datasets
    based on user-defined key and value columns with dynamic column mappings.
    Enhanced with improved matching logic and final summary DataFrame.
    """
    
    def __init__(self):
        self.left_df = None
        self.right_df = None
        self.left_keys = None
        self.right_keys = None
        self.left_values = None
        self.right_values = None
        self.column_mapping = None
        self.reconciliation_results = {}
        self.partial_match_threshold = 0.8  # Default threshold for partial matches
        self.value_weights = None  # Optional weights for value columns
        
    def reconcile_datasets(self, 
                      left_df: pd.DataFrame, 
                      right_df: pd.DataFrame,
                      left_key_columns: List[str],
                      right_key_columns: List[str],
                      left_value_columns: List[str],
                      right_value_columns: List[str],
                      column_mapping: Dict[str, str] = None,
                      manual_matches: List[Tuple[int, int]] = None,
                      partial_match_threshold: float = 0.8,
                      value_weights: List[float] = None,
                      combine_duplicates: bool = True,
                      left_name: str = "Left Dataset",
                      right_name: str = "Right Dataset") -> Dict[str, Any]:
        """
        Main reconciliation function that matches two datasets and provides comprehensive analysis.
        
        Parameters:
        -----------
        left_df : pd.DataFrame
            Left dataset to reconcile
        right_df : pd.DataFrame  
            Right dataset to reconcile
        left_key_columns : List[str]
            Primary key columns in left dataset
        right_key_columns : List[str]
            Primary key columns in right dataset
        left_value_columns : List[str]
            Value columns to compare in left dataset
        right_value_columns : List[str]
            Value columns to compare in right dataset
        column_mapping : Dict[str, str], optional
            Mapping between left and right column names
        manual_matches : List[Tuple[int, int]], optional
            Manual matches as list of (left_index, right_index) tuples
        partial_match_threshold : float, optional
            Threshold (0-1) for considering partial matches (default: 0.8)
        value_weights : List[float], optional
            Weights for each value column when calculating partial match scores
        combine_duplicates : bool, optional
            Whether to combine multiple entries with same keys (default: True)
            
        Returns:
        --------
        Dict containing all reconciliation results and DataFrames
        """
        
        # Store inputs
        self.left_df = left_df.copy().reset_index(drop=False)
        self.right_df = right_df.copy().reset_index(drop=False)
        # Sort both datasets by selected key columns
        self.left_df = self.left_df.sort_values(by=left_key_columns).reset_index(drop=True)
        self.right_df = self.right_df.sort_values(by=right_key_columns).reset_index(drop=True)
        self.left_keys = left_key_columns
        self.right_keys = right_key_columns
        self.left_values = left_value_columns
        self.right_values = right_value_columns
        self.column_mapping = column_mapping or {}
        self.partial_match_threshold = partial_match_threshold
        self.value_weights = value_weights
        self.combine_duplicates = combine_duplicates
        self.left_name = left_name
        self.right_name = right_name
        
        # Normalize weights if provided
        if self.value_weights is not None:
            if len(self.value_weights) != len(self.left_values):
                raise ValueError("Value weights must match length of value columns")
            # Normalize weights to sum to 1
            self.value_weights = [w/sum(self.value_weights) for w in self.value_weights]
        
        # Preserve original indexes
        if 'index' not in self.left_df.columns:
            self.left_df['original_index'] = self.left_df.index
        else:
            self.left_df['original_index'] = self.left_df['index']
            
        if 'index' not in self.right_df.columns:
            self.right_df['original_index'] = self.right_df.index
        else:
            self.right_df['original_index'] = self.right_df['index']
        
        # Initialize matched columns
        self.left_df['Matched'] = ''
        self.left_df['Match_Type'] = ''
        self.left_df['Match_Score'] = np.nan
        self.right_df['Matched'] = ''
        self.right_df['Match_Type'] = ''
        self.right_df['Match_Score'] = np.nan
        
        # Validate inputs
        self._validate_inputs()
        
        # Perform automated matching with enhanced logic
        self._perform_enhanced_matching()
        
        # Apply manual matches if provided
        if manual_matches:
            self._apply_manual_matches(manual_matches)
        
        # Generate reconciliation report
        results = self._generate_reconciliation_report()
        
        # Add final summary DataFrame
        results['final_summary_df'] = self._create_final_summary_dataframe()
        
        return results
    
    def _validate_inputs(self):
        """Validate input parameters and data consistency."""
        
        # Check if key columns exist
        missing_left_keys = [col for col in self.left_keys if col not in self.left_df.columns]
        missing_right_keys = [col for col in self.right_keys if col not in self.right_df.columns]
        
        if missing_left_keys:
            raise ValueError(f"Left key columns not found: {missing_left_keys}")
        if missing_right_keys:
            raise ValueError(f"Right key columns not found: {missing_right_keys}")
            
        # Check if value columns exist
        missing_left_values = [col for col in self.left_values if col not in self.left_df.columns]
        missing_right_values = [col for col in self.right_values if col not in self.right_df.columns]
        
        if missing_left_values:
            raise ValueError(f"Left value columns not found: {missing_left_values}")
        if missing_right_values:
            raise ValueError(f"Right value columns not found: {missing_right_values}")
            
        # Check if key and value column counts match
        if len(self.left_keys) != len(self.right_keys):
            raise ValueError("Number of key columns must match between left and right datasets")
        if len(self.left_values) != len(self.right_values):
            raise ValueError("Number of value columns must match between left and right datasets")
    
    def _create_composite_key(self, df: pd.DataFrame, key_columns: List[str]) -> pd.Series:
        """Create composite key from multiple columns."""
        if len(key_columns) == 1:
            return df[key_columns[0]].astype(str)
        else:
            return df[key_columns].astype(str).agg('|'.join, axis=1)
    
    def _perform_enhanced_matching(self):
        """
        Enhanced matching logic that handles exact matches with improved duplicate handling
        and supports combining entries with same keys.
        """
        
        # Create composite keys
        left_composite_keys = self._create_composite_key(self.left_df, self.left_keys)
        right_composite_keys = self._create_composite_key(self.right_df, self.right_keys)
        
        # Create mapping of keys to indexes for efficient lookup
        left_key_to_indexes = {}
        right_key_to_indexes = {}
        
        for idx, key in left_composite_keys.items():
            if key not in left_key_to_indexes:
                left_key_to_indexes[key] = []
            left_key_to_indexes[key].append(idx)
            
        for idx, key in right_composite_keys.items():
            if key not in right_key_to_indexes:
                right_key_to_indexes[key] = []
            right_key_to_indexes[key].append(idx)
        
        # Find matching keys
        matching_keys = set(left_key_to_indexes.keys()).intersection(set(right_key_to_indexes.keys()))
        
        # Process each matching key
        for key in matching_keys:
            left_indexes = left_key_to_indexes[key]
            right_indexes = right_key_to_indexes[key]
            
            if self.combine_duplicates and (len(left_indexes) > 1 or len(right_indexes) > 1):
                # Handle case where multiple entries need to be combined
                self._handle_duplicate_key_matching(key, left_indexes, right_indexes)
            else:
                # Standard matching logic for unique keys or when not combining
                self._handle_standard_key_matching(left_indexes, right_indexes)
    
    def _handle_standard_key_matching(self, left_indexes: List[int], right_indexes: List[int]):
        """Handle matching for keys with standard one-to-one or one-to-many logic."""
        
        # Track which right indexes have been matched to avoid double matching
        matched_right_indexes = set()
        
        for left_idx in left_indexes:
            left_row = self.left_df.loc[left_idx]
            best_matches = []
            
            for right_idx in right_indexes:
                if right_idx in matched_right_indexes:
                    continue
                    
                right_row = self.right_df.loc[right_idx]
                match_result = self._values_match(left_row, right_row)
                
                if match_result['exact_match']:
                    best_matches.append((right_idx, 'exact', 1.0))
                elif match_result['match_score'] >= self.partial_match_threshold:
                    best_matches.append((right_idx, 'partial', match_result['match_score']))
            
            # Sort by match quality (exact first, then by score)
            best_matches.sort(key=lambda x: (x[1] != 'exact', -x[2]))
            
            if best_matches:
                # Add all qualifying matches, but mark used right indexes
                matched_right_list = []
                for right_idx, match_type, score in best_matches:
                    matched_right_list.append(str(right_idx))
                    matched_right_indexes.add(right_idx)
                    
                    # Update right DataFrame with left index
                    self._add_single_match_to_df(self.right_df, right_idx, str(left_idx), match_type, score)
                
                # Update left DataFrame with all matching right indexes
                self._add_multiple_matches_to_df(self.left_df, left_idx, matched_right_list, 
                                               best_matches[0][1], best_matches[0][2])
    
    def _handle_duplicate_key_matching(self, key: str, left_indexes: List[int], right_indexes: List[int]):
        """
        Handle matching when there are multiple entries with the same key that should be combined.
        This creates aggregate matches and stores all related indexes.
        """
        
        # Calculate aggregate values for left side
        left_aggregates = {}
        for i, col in enumerate(self.left_values):
            if pd.api.types.is_numeric_dtype(self.left_df[col]):
                left_aggregates[col] = self.left_df.loc[left_indexes, col].sum()
            else:
                # For non-numeric, take the first non-null value
                non_null_values = self.left_df.loc[left_indexes, col].dropna()
                left_aggregates[col] = non_null_values.iloc[0] if len(non_null_values) > 0 else None
        
        # Calculate aggregate values for right side
        right_aggregates = {}
        for i, col in enumerate(self.right_values):
            if pd.api.types.is_numeric_dtype(self.right_df[col]):
                right_aggregates[col] = self.right_df.loc[right_indexes, col].sum()
            else:
                # For non-numeric, take the first non-null value
                non_null_values = self.right_df.loc[right_indexes, col].dropna()
                right_aggregates[col] = non_null_values.iloc[0] if len(non_null_values) > 0 else None
        
        # Compare aggregated values
        aggregate_match = self._compare_aggregated_values(left_aggregates, right_aggregates)
        
        # Store all cross-references
        right_indexes_str = [str(idx) for idx in right_indexes]
        left_indexes_str = [str(idx) for idx in left_indexes]
        
        # Update all left rows with all right indexes
        for left_idx in left_indexes:
            self._add_multiple_matches_to_df(self.left_df, left_idx, right_indexes_str, 
                                           aggregate_match['match_type'], aggregate_match['match_score'])
        
        # Update all right rows with all left indexes  
        for right_idx in right_indexes:
            self._add_multiple_matches_to_df(self.right_df, right_idx, left_indexes_str,
                                           aggregate_match['match_type'], aggregate_match['match_score'])
    
    def _compare_aggregated_values(self, left_aggregates: Dict, right_aggregates: Dict) -> Dict[str, Any]:
        """Compare aggregated values and return match information."""
        
        exact_match = True
        match_scores = []
        
        for i, (left_col, right_col) in enumerate(zip(self.left_values, self.right_values)):
            left_val = left_aggregates[left_col]
            right_val = right_aggregates[right_col]
            
            # Handle NaN values
            if pd.isna(left_val) and pd.isna(right_val):
                match_scores.append(1.0)
                continue
            elif pd.isna(left_val) or pd.isna(right_val):
                exact_match = False
                match_scores.append(0.0)
                continue
            
            # Calculate match score for this column
            col_score = 0.0
            
            try:
                if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                    if np.isclose(left_val, right_val, rtol=1e-9, atol=1e-9):
                        col_score = 1.0
                    else:
                        exact_match = False
                        # Calculate relative similarity for numeric values
                        max_val = max(abs(left_val), abs(right_val))
                        if max_val == 0:  # Both zero
                            col_score = 1.0
                        else:
                            col_score = max(0, 1 - (abs(left_val - right_val) / max_val))
                else:
                    # For strings or other types
                    left_str = str(left_val).strip().lower()
                    right_str = str(right_val).strip().lower()
                    
                    if left_str == right_str:
                        col_score = 1.0
                    else:
                        exact_match = False
                        # Use sequence matching for strings
                        if left_str and right_str:
                            # Simple ratio of common characters
                            common = set(left_str) & set(right_str)
                            col_score = len(common) / max(len(set(left_str)), len(set(right_str)))
            except:
                exact_match = False
                col_score = 0.0
            
            match_scores.append(col_score)
        
        # Calculate overall match score
        if self.value_weights:
            weighted_scores = [score * weight for score, weight in zip(match_scores, self.value_weights)]
            match_score = sum(weighted_scores)
        else:
            match_score = sum(match_scores) / len(match_scores) if match_scores else 0.0
        
        # Determine match type
        if exact_match:
            match_type = 'exact'
        elif match_score >= self.partial_match_threshold:
            match_type = 'partial'
        else:
            match_type = 'no_match'
        
        return {
            'exact_match': exact_match,
            'match_score': match_score,
            'match_type': match_type
        }
    
    def _add_single_match_to_df(self, df: pd.DataFrame, row_idx: int, match_idx: str, 
                               match_type: str, match_score: float):
        """Add a single match to a DataFrame row."""
        
        current_matches = str(df.at[row_idx, 'Matched'])
        if current_matches == '' or pd.isna(current_matches) or current_matches == 'nan':
            df.at[row_idx, 'Matched'] = match_idx
            df.at[row_idx, 'Match_Type'] = match_type
            df.at[row_idx, 'Match_Score'] = match_score
        else:
            existing_matches = current_matches.split(',')
            if match_idx not in existing_matches:
                df.at[row_idx, 'Matched'] = current_matches + ',' + match_idx
                # For multiple matches, keep the best match type and score
                if (match_score > df.at[row_idx, 'Match_Score'] or 
                    pd.isna(df.at[row_idx, 'Match_Score']) or
                    (match_type == 'exact' and df.at[row_idx, 'Match_Type'] != 'exact')):
                    df.at[row_idx, 'Match_Type'] = match_type
                    df.at[row_idx, 'Match_Score'] = match_score
    
    def _add_multiple_matches_to_df(self, df: pd.DataFrame, row_idx: int, match_indexes: List[str],
                                   match_type: str, match_score: float):
        """Add multiple matches to a DataFrame row."""
        
        current_matches = str(df.at[row_idx, 'Matched'])
        if current_matches == '' or pd.isna(current_matches) or current_matches == 'nan':
            df.at[row_idx, 'Matched'] = ','.join(match_indexes)
        else:
            existing_matches = set(current_matches.split(','))
            all_matches = existing_matches.union(set(match_indexes))
            df.at[row_idx, 'Matched'] = ','.join(sorted(all_matches, key=int))
        
        df.at[row_idx, 'Match_Type'] = match_type
        df.at[row_idx, 'Match_Score'] = match_score
    
    def _values_match(self, left_row: pd.Series, right_row: pd.Series) -> Dict[str, Union[bool, float]]:
        """
        Check if value columns match between two rows.
        Returns dict with:
        - exact_match: boolean indicating if all values match exactly
        - match_score: float (0-1) representing overall match quality
        """
        exact_match = True
        match_scores = []
        
        for i, (left_col, right_col) in enumerate(zip(self.left_values, self.right_values)):
            left_val = left_row[left_col]
            right_val = right_row[right_col]
            
            # Handle NaN values
            if pd.isna(left_val) and pd.isna(right_val):
                match_scores.append(1.0)  # Consider matching NaNs as full match
                continue
            elif pd.isna(left_val) or pd.isna(right_val):
                exact_match = False
                match_scores.append(0.0)
                continue
            
            # Calculate match score for this column
            col_score = 0.0
            
            try:
                if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                    if np.isclose(left_val, right_val, rtol=1e-9, atol=1e-9):
                        col_score = 1.0
                    else:
                        exact_match = False
                        # Calculate relative similarity for numeric values
                        max_val = max(abs(left_val), abs(right_val))
                        if max_val == 0:  # Both zero
                            col_score = 1.0
                        else:
                            col_score = max(0, 1 - (abs(left_val - right_val) / max_val))
                else:
                    # For strings or other types
                    left_str = str(left_val).strip().lower()
                    right_str = str(right_val).strip().lower()
                    
                    if left_str == right_str:
                        col_score = 1.0
                    else:
                        exact_match = False
                        # Use sequence matching for strings
                        if left_str and right_str:
                            # Simple ratio of common characters
                            common = set(left_str) & set(right_str)
                            col_score = len(common) / max(len(set(left_str)), len(set(right_str)))
            except:
                exact_match = False
                col_score = 0.0
            
            match_scores.append(col_score)
        
        # Calculate overall match score
        if self.value_weights:
            weighted_scores = [score * weight for score, weight in zip(match_scores, self.value_weights)]
            match_score = sum(weighted_scores)
        else:
            match_score = sum(match_scores) / len(match_scores) if match_scores else 0.0
        
        return {
            'exact_match': exact_match,
            'match_score': match_score
        }
    
    def _add_match(self, left_idx: int, right_idx: int, match_type: str = 'exact', match_score: float = 1.0):
        """Add match between left and right rows (legacy method for compatibility)."""
        self._add_single_match_to_df(self.left_df, left_idx, str(right_idx), match_type, match_score)
        self._add_single_match_to_df(self.right_df, right_idx, str(left_idx), match_type, match_score)
    
    def _apply_manual_matches(self, manual_matches: List[Tuple[int, int]]):
        """Apply manual matches provided by user."""
        
        for left_idx, right_idx in manual_matches:
            # Validate indexes exist
            if left_idx not in self.left_df.index:
                warnings.warn(f"Left index {left_idx} not found in dataset")
                continue
            if right_idx not in self.right_df.index:
                warnings.warn(f"Right index {right_idx} not found in dataset")
                continue
            
            # Add manual match
            self._add_match(left_idx, right_idx, 'manual', 1.0)
    
    def _create_final_summary_dataframe(self) -> pd.DataFrame:
        """
        Create a final summary DataFrame that consolidates all reconciliation results.
        """
        
        summary_data = []
        
        # Get all unique composite keys from both datasets
        left_composite_keys = self._create_composite_key(self.left_df, self.left_keys)
        right_composite_keys = self._create_composite_key(self.right_df, self.right_keys)
        
        all_keys = set(left_composite_keys.unique()).union(set(right_composite_keys.unique()))
        
        for key in all_keys:
            # Get rows with this key from both datasets
            left_mask = left_composite_keys == key
            right_mask = right_composite_keys == key
            
            left_rows = self.left_df[left_mask]
            right_rows = self.right_df[right_mask]
            
            # Determine reconciliation status
            status = self._determine_reconciliation_status(left_rows, right_rows)
            
            # Calculate aggregated values
            left_totals = {}
            right_totals = {}
            
            for i, (left_col, right_col) in enumerate(zip(self.left_values, self.right_values)):
                if len(left_rows) > 0:
                    if pd.api.types.is_numeric_dtype(left_rows[left_col]):
                        left_totals[left_col] = left_rows[left_col].sum()
                    else:
                        left_totals[left_col] = left_rows[left_col].iloc[0] if not left_rows[left_col].isna().all() else None
                else:
                    left_totals[left_col] = 0 if any(pd.api.types.is_numeric_dtype(self.left_df[col]) for col in self.left_values) else None
                
                if len(right_rows) > 0:
                    if pd.api.types.is_numeric_dtype(right_rows[right_col]):
                        right_totals[right_col] = right_rows[right_col].sum()
                    else:
                        right_totals[right_col] = right_rows[right_col].iloc[0] if not right_rows[right_col].isna().all() else None
                else:
                    right_totals[right_col] = 0 if any(pd.api.types.is_numeric_dtype(self.right_df[col]) for col in self.right_values) else None
            
            # Create summary row with dataset-specific names
            summary_row = {
                'Key': key,
                'Status': status,
                f'{self.left_name}_Row_Count': len(left_rows),
                f'{self.right_name}_Row_Count': len(right_rows),
                f'{self.left_name}_Indexes': ','.join(map(str, left_rows.index.tolist())) if len(left_rows) > 0 else '',
                f'{self.right_name}_Indexes': ','.join(map(str, right_rows.index.tolist())) if len(right_rows) > 0 else '',
            }
            
            # Add key columns
            if len(left_rows) > 0:
                for key_col in self.left_keys:
                    summary_row[f'Key_{key_col}'] = left_rows[key_col].iloc[0]
            elif len(right_rows) > 0:
                for i, key_col in enumerate(self.right_keys):
                    summary_row[f'Key_{self.left_keys[i]}'] = right_rows[key_col].iloc[0]
            
            # Add value columns with dataset-specific names
            for i, (left_col, right_col) in enumerate(zip(self.left_values, self.right_values)):
                summary_row[f'{self.left_name}_{left_col}'] = left_totals[left_col]
                summary_row[f'{self.right_name}_{right_col}'] = right_totals[right_col]
                
                # Calculate difference for numeric columns
                if (pd.api.types.is_numeric_dtype(self.left_df[left_col]) and 
                    pd.api.types.is_numeric_dtype(self.right_df[right_col])):
                    left_val = left_totals[left_col] if left_totals[left_col] is not None else 0
                    right_val = right_totals[right_col] if right_totals[right_col] is not None else 0
                    summary_row[f'Diff_{left_col}_{right_col}'] = left_val - right_val
            
            # Add match information
            if len(left_rows) > 0:
                match_types = left_rows['Match_Type'].dropna().unique()
                match_scores = left_rows['Match_Score'].dropna()
                summary_row['Match_Type'] = ','.join(match_types) if len(match_types) > 0 else 'No Match'
                summary_row['Avg_Match_Score'] = match_scores.mean() if len(match_scores) > 0 else 0.0
            else:
                summary_row['Match_Type'] = f'{self.right_name} Only'
                summary_row['Avg_Match_Score'] = 0.0
            
            summary_data.append(summary_row)
        
        # Create DataFrame and sort by status and key
        final_summary_df = pd.DataFrame(summary_data)
        
        if not final_summary_df.empty:
            # Sort by status priority and then by key
            status_order = {'Matched': 1, 'Partial': 2, 'Mismatched': 3, 
                        f'{self.left_name} Only': 4, f'{self.right_name} Only': 5}
            final_summary_df['Status_Order'] = final_summary_df['Status'].map(status_order)
            final_summary_df = final_summary_df.sort_values(['Status_Order', 'Key']).drop('Status_Order', axis=1)
        
        return final_summary_df

    def _determine_reconciliation_status(self, left_rows: pd.DataFrame, right_rows: pd.DataFrame) -> str:
        """Determine the reconciliation status for a given key."""
        
        if len(left_rows) == 0:
            return f'{self.right_name} Only'
        elif len(right_rows) == 0:
            return f'{self.left_name} Only'
        else:
            # Check match types in left rows
            match_types = left_rows['Match_Type'].dropna().unique()
            
            if len(match_types) == 0 or all(mt == '' for mt in match_types):
                return 'Mismatched'
            elif 'exact' in match_types or 'manual' in match_types:
                return 'Matched'
            elif 'partial' in match_types:
                return 'Partial'
            else:
                return 'Mismatched'
    
    def _generate_reconciliation_report(self) -> Dict[str, Any]:
        """Generate comprehensive reconciliation report."""
        
        # Categorize entries
        matching_entries = self._get_matching_entries()
        left_only_entries = self._get_left_only_entries()
        right_only_entries = self._get_right_only_entries()
        mismatched_entries = self._get_mismatched_entries()
        partial_matches = self._get_partial_matches()
        
        # Create tallied and unmatched DataFrames
        tallied_entries = matching_entries.copy()
        
        # Combine unmatched entries with dataset-specific source labels
        unmatched_entries = pd.concat([
            left_only_entries.assign(Source=f'{self.left_name}_Only'),
            right_only_entries.assign(Source=f'{self.right_name}_Only'),
            mismatched_entries.assign(Source='Mismatched'),
            partial_matches.assign(Source='Partial_Match')
        ], ignore_index=True)
        
        # Calculate totals
        totals_summary = self._calculate_totals(
            matching_entries, left_only_entries, right_only_entries, 
            mismatched_entries, partial_matches
        )
        
        # Create reconciliation statement
        reconciliation_statement = self._create_reconciliation_statement(totals_summary)
        
        # Prepare final results
        results = {
            'left_df_updated': self.left_df,
            'right_df_updated': self.right_df,
            'matching_entries': matching_entries,
            'left_only_entries': left_only_entries,
            'right_only_entries': right_only_entries,
            'mismatched_entries': mismatched_entries,
            'partial_matches': partial_matches,
            'tallied_entries': tallied_entries,
            'unmatched_entries': unmatched_entries,
            'totals_summary': totals_summary,
            'reconciliation_statement': reconciliation_statement,
            'summary_stats': {
                f'total_{self.left_name.lower().replace(" ", "_")}_records': len(self.left_df),
                f'total_{self.right_name.lower().replace(" ", "_")}_records': len(self.right_df),
                'matching_records': len(matching_entries),
                f'{self.left_name.lower().replace(" ", "_")}_only_records': len(left_only_entries),
                f'{self.right_name.lower().replace(" ", "_")}_only_records': len(right_only_entries),
                'mismatched_records': len(mismatched_entries),
                'partial_match_records': len(partial_matches),
                'reconciliation_rate': len(matching_entries) / max(len(self.left_df), 1) * 100,
                'partial_reconciliation_rate': (len(matching_entries) + len(partial_matches)) / max(len(self.left_df), 1) * 100
            }
        }
        
        return results
    
    def _get_matching_entries(self) -> pd.DataFrame:
        """Get all entries that have exact matches."""
        
        matching_left = self.left_df[(self.left_df['Match_Type'] == 'exact') | 
                                   (self.left_df['Match_Type'] == 'manual')].copy()
        matching_right = self.right_df[(self.right_df['Match_Type'] == 'exact') | 
                                     (self.right_df['Match_Type'] == 'manual')].copy()
        
        if len(matching_left) == 0:
            return pd.DataFrame()
        
        # Add source identifier
        matching_left['Source'] = 'Left'
        matching_right['Source'] = 'Right'
        
        # Combine matching entries
        matching_entries = pd.concat([matching_left, matching_right], ignore_index=True)
        
        return matching_entries
    
    def _get_partial_matches(self) -> pd.DataFrame:
        """Get all entries that have partial matches."""
        
        partial_left = self.left_df[self.left_df['Match_Type'] == 'partial'].copy()
        partial_right = self.right_df[self.right_df['Match_Type'] == 'partial'].copy()
        
        if len(partial_left) == 0:
            return pd.DataFrame()
        
        # Add source identifier
        partial_left['Source'] = 'Left_Partial'
        partial_right['Source'] = 'Right_Partial'
        
        # Combine partial matches
        partial_entries = pd.concat([partial_left, partial_right], ignore_index=True)
        
        return partial_entries
    
    def _get_left_only_entries(self) -> pd.DataFrame:
        """Get entries that exist only in left dataset."""
        
        left_only = self.left_df[self.left_df['Matched'] == ''].copy()
        
        # Check if these are truly unmatched or just mismatched
        left_composite_keys = self._create_composite_key(left_only, self.left_keys)
        right_composite_keys = self._create_composite_key(self.right_df, self.right_keys)
        
        # Filter out entries that have key matches in right (these are mismatched, not left-only)
        truly_left_only = left_only[~left_composite_keys.isin(right_composite_keys)]
        
        return truly_left_only
    
    def _get_right_only_entries(self) -> pd.DataFrame:
        """Get entries that exist only in right dataset."""
        
        right_only = self.right_df[self.right_df['Matched'] == ''].copy()
        
        # Check if these are truly unmatched or just mismatched
        right_composite_keys = self._create_composite_key(right_only, self.right_keys)
        left_composite_keys = self._create_composite_key(self.left_df, self.left_keys)
        
        # Filter out entries that have key matches in left (these are mismatched, not right-only)
        truly_right_only = right_only[~right_composite_keys.isin(left_composite_keys)]
        
        return truly_right_only
    
    def _get_mismatched_entries(self) -> pd.DataFrame:
        """Get entries where keys match but values don't."""
        
        mismatched_entries = []
        
        # Get unmatched entries from both sides (excluding partial matches)
        left_unmatched = self.left_df[(self.left_df['Matched'] == '') & 
                                    (self.left_df['Match_Type'] != 'partial')]
        right_unmatched = self.right_df[(self.right_df['Matched'] == '') & 
                                      (self.right_df['Match_Type'] != 'partial')]
        
        if len(left_unmatched) == 0 or len(right_unmatched) == 0:
            return pd.DataFrame()
        
        # Create composite keys
        left_keys = self._create_composite_key(left_unmatched, self.left_keys)
        right_keys = self._create_composite_key(right_unmatched, self.right_keys)
        
        # Find keys that exist in both but weren't matched (meaning values differ)
        common_keys = set(left_keys).intersection(set(right_keys))
        
        for key in common_keys:
            # Get rows with this key from both sides
            left_matches = left_unmatched[left_keys == key]
            right_matches = right_unmatched[right_keys == key]
            
            # Add to mismatched (these have matching keys but different values)
            left_with_source = left_matches.copy()
            left_with_source['Source'] = 'Left_Mismatched'
            
            right_with_source = right_matches.copy()
            right_with_source['Source'] = 'Right_Mismatched'
            
            mismatched_entries.append(left_with_source)
            mismatched_entries.append(right_with_source)
        
        if mismatched_entries:
            return pd.concat(mismatched_entries, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _calculate_totals(self, matching_entries, left_only_entries, 
                         right_only_entries, mismatched_entries, partial_matches) -> Dict[str, Any]:
        """Calculate totals for all value columns across different categories."""
        
        totals = {}
        
        # Calculate totals for each value column
        for i, (left_col, right_col) in enumerate(zip(self.left_values, self.right_values)):
            col_totals = {
                'column_name': f"{left_col}_{right_col}",
                'left_total': self.left_df[left_col].sum() if pd.api.types.is_numeric_dtype(self.left_df[left_col]) else 0,
                'right_total': self.right_df[right_col].sum() if pd.api.types.is_numeric_dtype(self.right_df[right_col]) else 0,
                'matching_left_total': 0,
                'matching_right_total': 0,
                'left_only_total': 0,
                'right_only_total': 0,
                'mismatched_left_total': 0,
                'mismatched_right_total': 0,
                'partial_left_total': 0,
                'partial_right_total': 0
            }
            
            # Calculate matching totals
            if len(matching_entries) > 0:
                matching_left = matching_entries[matching_entries['Source'] == 'Left']
                matching_right = matching_entries[matching_entries['Source'] == 'Right']
                
                if len(matching_left) > 0 and left_col in matching_left.columns:
                    if pd.api.types.is_numeric_dtype(matching_left[left_col]):
                        col_totals['matching_left_total'] = matching_left[left_col].sum()
                
                if len(matching_right) > 0 and right_col in matching_right.columns:
                    if pd.api.types.is_numeric_dtype(matching_right[right_col]):
                        col_totals['matching_right_total'] = matching_right[right_col].sum()
            
            # Calculate left-only totals
            if len(left_only_entries) > 0 and left_col in left_only_entries.columns:
                if pd.api.types.is_numeric_dtype(left_only_entries[left_col]):
                    col_totals['left_only_total'] = left_only_entries[left_col].sum()
            
            # Calculate right-only totals
            if len(right_only_entries) > 0 and right_col in right_only_entries.columns:
                if pd.api.types.is_numeric_dtype(right_only_entries[right_col]):
                    col_totals['right_only_total'] = right_only_entries[right_col].sum()
            
            # Calculate mismatched totals
            if len(mismatched_entries) > 0:
                mismatched_left = mismatched_entries[mismatched_entries['Source'] == 'Left_Mismatched']
                mismatched_right = mismatched_entries[mismatched_entries['Source'] == 'Right_Mismatched']
                
                if len(mismatched_left) > 0 and left_col in mismatched_left.columns:
                    if pd.api.types.is_numeric_dtype(mismatched_left[left_col]):
                        col_totals['mismatched_left_total'] = mismatched_left[left_col].sum()
                
                if len(mismatched_right) > 0 and right_col in mismatched_right.columns:
                    if pd.api.types.is_numeric_dtype(mismatched_right[right_col]):
                        col_totals['mismatched_right_total'] = mismatched_right[right_col].sum()
            
            # Calculate partial match totals
            if len(partial_matches) > 0:
                partial_left = partial_matches[partial_matches['Source'] == 'Left_Partial']
                partial_right = partial_matches[partial_matches['Source'] == 'Right_Partial']
                
                if len(partial_left) > 0 and left_col in partial_left.columns:
                    if pd.api.types.is_numeric_dtype(partial_left[left_col]):
                        col_totals['partial_left_total'] = partial_left[left_col].sum()
                
                if len(partial_right) > 0 and right_col in partial_right.columns:
                    if pd.api.types.is_numeric_dtype(partial_right[right_col]):
                        col_totals['partial_right_total'] = partial_right[right_col].sum()
            
            # Calculate difference
            col_totals['total_difference'] = col_totals['left_total'] - col_totals['right_total']
            
            totals[f"column_{i+1}"] = col_totals
        
        return totals
    
    def _create_reconciliation_statement(self, totals_summary: Dict[str, Any]) -> str:
        """Create a comprehensive reconciliation statement."""
        
        statement = []
        statement.append("="*60)
        statement.append("RECONCILIATION STATEMENT")
        statement.append("="*60)
        statement.append("")
        
        # Dataset summary
        statement.append(f"{self.left_name} Records: {len(self.left_df):,}")
        statement.append(f"{self.right_name} Records: {len(self.right_df):,}")
        statement.append("")
        
        # Reconciliation summary
        matching_count = len(self._get_matching_entries()) // 2  # Divide by 2 since we count both sides
        partial_count = len(self._get_partial_matches()) // 2
        left_only_count = len(self._get_left_only_entries())
        right_only_count = len(self._get_right_only_entries())
        mismatched_count = len(self._get_mismatched_entries()) // 2  # Divide by 2 since we count both sides
        
        statement.append("RECONCILIATION BREAKDOWN:")
        statement.append(f"  Fully Matched Records: {matching_count:,}")
        statement.append(f"  Partial Matches (>={self.partial_match_threshold*100:.0f}%): {partial_count:,}")
        statement.append(f"  {self.left_name} Only Records: {left_only_count:,}")
        statement.append(f"  {self.right_name} Only Records: {right_only_count:,}")
        statement.append(f"  Mismatched Records: {mismatched_count:,}")
        statement.append("")
        # Enhanced features info
        if self.combine_duplicates:
            statement.append("ENHANCED FEATURES:")
            statement.append("  ✓ Duplicate key combination enabled")
            statement.append("  ✓ Enhanced matching with comma-separated indexes")
            statement.append("")
        
        # Value column analysis
        for col_key, col_totals in totals_summary.items():
            statement.append(f"ANALYSIS FOR {col_totals['column_name']}:")
            statement.append(f"  Left Total: {col_totals['left_total']:,.2f}")
            statement.append(f"  Right Total: {col_totals['right_total']:,.2f}")
            statement.append(f"  Difference: {col_totals['total_difference']:,.2f}")
            statement.append("")
            statement.append("  Breakdown:")
            statement.append(f"    Matched Left: {col_totals['matching_left_total']:,.2f}")
            statement.append(f"    Matched Right: {col_totals['matching_right_total']:,.2f}")
            statement.append(f"    Partial Left: {col_totals['partial_left_total']:,.2f}")
            statement.append(f"    Partial Right: {col_totals['partial_right_total']:,.2f}")
            statement.append(f"    Left Only: {col_totals['left_only_total']:,.2f}")
            statement.append(f"    Right Only: {col_totals['right_only_total']:,.2f}")
            statement.append(f"    Mismatched Left: {col_totals['mismatched_left_total']:,.2f}")
            statement.append(f"    Mismatched Right: {col_totals['mismatched_right_total']:,.2f}")
            statement.append("")
        
        # Reconciliation verification
        statement.append("RECONCILIATION VERIFICATION:")
        all_verified = True
        for col_key, col_totals in totals_summary.items():
            calculated_diff = (col_totals['matching_left_total'] + col_totals['partial_left_total'] + 
                             col_totals['left_only_total'] + col_totals['mismatched_left_total']) - \
                            (col_totals['matching_right_total'] + col_totals['partial_right_total'] + 
                             col_totals['right_only_total'] + col_totals['mismatched_right_total'])
            verification_status = "✓ VERIFIED" if abs(calculated_diff - col_totals['total_difference']) < 0.01 else "✗ ERROR"
            statement.append(f"  {col_totals['column_name']}: {verification_status}")
            if abs(calculated_diff - col_totals['total_difference']) >= 0.01:
                all_verified = False
        
        statement.append("")
        statement.append(f"Overall Reconciliation Status: {'✓ SUCCESSFUL' if all_verified else '✗ DISCREPANCIES FOUND'}")
        statement.append("="*60)
        
        return "\n".join(statement)
    def manual_reconcile(self, left_indexes: List[int], right_indexes: List[int], match_type: str = 'manual') -> Dict[str, Any]:
        """
        Perform manual reconciliation between selected left and right rows.
        
        Parameters:
        -----------
        left_indexes : List[int]
            List of indexes from left dataset to reconcile
        right_indexes : List[int]
            List of indexes from right dataset to reconcile
        match_type : str, optional
            Type of match ('manual', 'exact', 'partial') (default: 'manual')
            
        Returns:
        --------
        Dict containing reconciliation results and updated dataframes
        """
        # Validate indexes
        invalid_left = [idx for idx in left_indexes if idx not in self.left_df.index]
        invalid_right = [idx for idx in right_indexes if idx not in self.right_df.index]
        
        if invalid_left:
            raise ValueError(f"Invalid left indexes: {invalid_left}")
        if invalid_right:
            raise ValueError(f"Invalid right indexes: {invalid_right}")
        
        # Calculate match score (1.0 for manual exact, 0.8 for manual partial)
        match_score = 1.0 if match_type == 'exact' else 0.8
        
        # Create string representations of matched indexes
        left_indexes_str = [str(idx) for idx in left_indexes]
        right_indexes_str = [str(idx) for idx in right_indexes]
        
        # Update all left rows with all right indexes
        for left_idx in left_indexes:
            self._add_multiple_matches_to_df(self.left_df, left_idx, right_indexes_str, match_type, match_score)
        
        # Update all right rows with all left indexes
        for right_idx in right_indexes:
            self._add_multiple_matches_to_df(self.right_df, right_idx, left_indexes_str, match_type, match_score)
        
        # Regenerate the reconciliation report
        results = self._generate_reconciliation_report()
        results['final_summary_df'] = self._create_final_summary_dataframe()
        
        return results
    
    def validate_manual_matches(self, manual_matches: List[Tuple[int, int]]) -> Dict[str, Any]:
            """
            Validate manual matches before applying them.
            
            Parameters:
            -----------
            manual_matches : List[Tuple[int, int]]
                List of (left_index, right_index) tuples to validate
                
            Returns:
            --------
            Dict with validation results
            """
            validation_results = {
                'valid_matches': [],
                'invalid_matches': [],
                'warnings': [],
                'totals_match': True,
                'total_differences': {}
            }
            
            for left_idx, right_idx in manual_matches:
                # Check if indexes exist
                if left_idx not in self.left_df.index:
                    validation_results['invalid_matches'].append((left_idx, right_idx, "Left index not found"))
                    continue
                if right_idx not in self.right_df.index:
                    validation_results['invalid_matches'].append((left_idx, right_idx, "Right index not found"))
                    continue
                
                # Check if already matched
                if str(self.left_df.at[left_idx, 'Matched']) != '':
                    validation_results['warnings'].append(f"Left index {left_idx} already has matches")
                if str(self.right_df.at[right_idx, 'Matched']) != '':
                    validation_results['warnings'].append(f"Right index {right_idx} already has matches")
                
                validation_results['valid_matches'].append((left_idx, right_idx))
            
            # Check if totals match for value columns
            if validation_results['valid_matches']:
                for i, (left_col, right_col) in enumerate(zip(self.left_values, self.right_values)):
                    left_total = sum([self.left_df.at[left_idx, left_col] for left_idx, _ in validation_results['valid_matches'] 
                                    if pd.api.types.is_numeric_dtype(self.left_df[left_col])])
                    right_total = sum([self.right_df.at[right_idx, right_col] for _, right_idx in validation_results['valid_matches']
                                    if pd.api.types.is_numeric_dtype(self.right_df[right_col])])
                    
                    difference = left_total - right_total
                    validation_results['total_differences'][f"{left_col}_{right_col}"] = {
                        'left_total': left_total,
                        'right_total': right_total,
                        'difference': difference
                    }
                    
                    if abs(difference) > 0.01:  # Allow for small floating point differences
                        validation_results['totals_match'] = False
            
            return validation_results

    def export_results(self, results: Dict[str, Any], export_path: str = None) -> Dict[str, str]:
        """
        Export reconciliation results to multiple files.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Results dictionary from reconcile_datasets()
        export_path : str, optional
            Base path for export files (default: current directory)
            
        Returns:
        --------
        Dict with paths of exported files
        """
        if export_path is None:
          export_path = ""
    
        exported_files = {}
    
        # Clean dataset names for filenames
        left_name_clean = self.left_name.lower().replace(" ", "_")
        right_name_clean = self.right_name.lower().replace(" ", "_")
        
        # Export main results
        results['left_df_updated'].to_csv(f"{export_path}{left_name_clean}_updated.csv", index=False)
        exported_files['left_updated'] = f"{export_path}{left_name_clean}_updated.csv"
        
        results['right_df_updated'].to_csv(f"{export_path}{right_name_clean}_updated.csv", index=False)
        exported_files['right_updated'] = f"{export_path}{right_name_clean}_updated.csv"
        
        # Export final summary
        if not results['final_summary_df'].empty:
            results['final_summary_df'].to_csv(f"{export_path}final_summary.csv", index=False)
            exported_files['final_summary'] = f"{export_path}final_summary.csv"
        
        # Export reconciliation statement
        with open(f"{export_path}reconciliation_statement.txt", 'w') as f:
            f.write(results['reconciliation_statement'])
        exported_files['reconciliation_statement'] = f"{export_path}reconciliation_statement.txt"
        
        # Export other categorized results if they exist
        for key in ['matching_entries', 'left_only_entries', 'right_only_entries', 
                   'mismatched_entries', 'partial_matches', 'unmatched_entries']:
            if key in results and not results[key].empty:
                results[key].to_csv(f"{export_path}{key}.csv", index=False)
                exported_files[key] = f"{export_path}{key}.csv"
        
        return exported_files

# Example usage and demonstration
def example_usage():
    """
    Example demonstrating the enhanced reconciliation engine with improved matching logic.
    """
    
    # Create sample data with duplicates to demonstrate enhanced matching
    left_data = {
        'ID': ['A001', 'A001', 'A002', 'A003', 'A004'],
        'Name': ['Product A', 'Product A', 'Product B', 'Product C', 'Product D'],
        'Amount': [100, 50, 200, 150, 75],
        'Quantity': [10, 5, 20, 15, 8]
    }
    
    right_data = {
        'ProductID': ['A001', 'A001', 'A002', 'A005', 'A006'],
        'ProductName': ['Product A', 'Product A', 'Product B', 'Product E', 'Product F'],
        'Value': [150, 0, 200, 100, 90],  # Combined A001 total = 150
        'Count': [15, 0, 20, 12, 9]       # Combined A001 total = 15
    }
    
    left_df = pd.DataFrame(left_data)
    right_df = pd.DataFrame(right_data)
    
    print("Sample Left DataFrame:")
    print(left_df)
    print("\nSample Right DataFrame:")
    print(right_df)
    
    # Initialize reconciliation engine
    engine = DataReconciliationEngine()
    
    # Perform reconciliation with enhanced features
    results = engine.reconcile_datasets(
        left_df=left_df,
        right_df=right_df,
        left_key_columns=['ID'],
        right_key_columns=['ProductID'],
        left_value_columns=['Amount', 'Quantity'],
        right_value_columns=['Value', 'Count'],
        combine_duplicates=True,  # Enable duplicate combination
        partial_match_threshold=0.9
    )
    
    print("\n" + "="*60)
    print("RECONCILIATION RESULTS")
    print("="*60)
    
    print("\nLeft DataFrame with Matches:")
    print(results['left_df_updated'][['ID', 'Name', 'Amount', 'Quantity', 'Matched', 'Match_Type', 'Match_Score']])
    
    print("\nRight DataFrame with Matches:")
    print(results['right_df_updated'][['ProductID', 'ProductName', 'Value', 'Count', 'Matched', 'Match_Type', 'Match_Score']])
    
    print("\nFinal Summary DataFrame:")
    print(results['final_summary_df'])
    
    print("\n" + results['reconciliation_statement'])
    
    return results

