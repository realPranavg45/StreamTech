"""
data_diff.py

Backend functions for comparing two tabular datasets (pandas DataFrames)
by primary key(s). No main() in this file — only functions.

Functions:
- compare_dataframes(old_df, new_df, pk_cols) -> dict with keys:
    'new_rows', 'deleted_rows', 'modified_merged', 'modified_keys'
- make_diff_styler(merged_df, pk_cols) -> pandas.Styler highlighting changed values
- make_diff_html(merged_df, pk_cols) -> HTML string with inline highlights
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import io
from pandas.io.formats.style import Styler


def _ensure_pk_in_df(df: pd.DataFrame, pk_cols: List[str]):
    missing = [c for c in pk_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Primary key column(s) not found in DataFrame: {missing}")

def compare_dataframes(old_df: pd.DataFrame, new_df: pd.DataFrame, pk_cols: List[str]) -> Dict[str, Any]:
    """
    Compare old_df and new_df using pk_cols (one or more columns).

    Returns dict:
      - 'new_rows': DataFrame of rows present only in new_df
      - 'deleted_rows': DataFrame of rows present only in old_df
      - 'modified_merged': DataFrame merged on pk with suffixes _old/_new for columns (only rows where any non-PK column changed)
      - 'modified_keys': list of primary key tuples which were modified
    """
    if not isinstance(pk_cols, (list, tuple)):
        raise ValueError("pk_cols must be a list or tuple of column names.")
    pk_cols = list(pk_cols)

    _ensure_pk_in_df(old_df, pk_cols)
    _ensure_pk_in_df(new_df, pk_cols)

    # Use a copy so we don't mutate original frames.
    old = old_df.copy().reset_index(drop=True)
    new = new_df.copy().reset_index(drop=True)

    # Normalize index by PK to simplify set operations — but keep a column-level approach
    # Create a unique key tuple column for set difference (works even with multiple PKs)
    def make_key_series(df):
        if len(pk_cols) == 1:
            return df[pk_cols[0]].astype(object)
        # for multi-column PK, make tuples (preserve order)
        return df[pk_cols].apply(lambda row: tuple(row.values), axis=1)

    old_keys = pd.Series(make_key_series(old), index=old.index)
    new_keys = pd.Series(make_key_series(new), index=new.index)

    # Map key -> index
    old_key_to_idx = dict(zip(old_keys, old.index))
    new_key_to_idx = dict(zip(new_keys, new.index))

    old_key_set = set(old_keys)
    new_key_set = set(new_keys)

    # Keys for new / deleted
    added_keys = new_key_set - old_key_set
    deleted_keys = old_key_set - new_key_set
    common_keys = old_key_set & new_key_set

    # new_rows and deleted_rows as DataFrames
    if added_keys:
        # new_keys is a Series of keys aligned with new's index
        new_mask = new_keys.isin(added_keys)
        new_rows = new.loc[new_mask].reset_index(drop=True)
    else:
        new_rows = pd.DataFrame()

    if deleted_keys:
        old_mask = old_keys.isin(deleted_keys)
        deleted_rows = old.loc[old_mask].reset_index(drop=True)
    else:
        deleted_rows = pd.DataFrame()
    # For modified rows: merge old and new on PK(s) with suffixes, then check non-PK columns for inequality.
    # Use an "outer" merge restricted to common PKs by building filtered frames with only common keys.
    if common_keys:
        old_common = old.loc[[old_key_to_idx[k] for k in sorted(common_keys, key=lambda x: str(x))]].copy()
        new_common = new.loc[[new_key_to_idx[k] for k in sorted(common_keys, key=lambda x: str(x))]].copy()
        # Reset index to ensure merge works from PK columns
        old_common = old_common.reset_index(drop=True)
        new_common = new_common.reset_index(drop=True)
        merged = pd.merge(old_common, new_common, on=pk_cols, how='inner', suffixes=('_old', '_new'), sort=False)
        # Determine which non-PK columns changed
        non_pk_cols = [c for c in old.columns if c not in pk_cols]
        changed_mask = pd.Series(False, index=merged.index)
        for col in non_pk_cols:
            old_col = col + '_old'
            new_col = col + '_new'
            # Compare values while treating NaNs as equal
            neq = ~merged[old_col].combine(merged[new_col], lambda a, b: (pd.isna(a) and pd.isna(b)) or a == b)
            # The combine above might return True/False/NA; coerce to boolean
            neq = neq.astype(bool)
            changed_mask = changed_mask | neq

        modified_merged = merged[changed_mask].reset_index(drop=True)
        # build modified_keys as list of pk tuples (or scalar if single pk)
        if len(pk_cols) == 1:
            modified_keys = modified_merged[pk_cols[0]].tolist()
        else:
            modified_keys = [tuple(row) for row in modified_merged[pk_cols].values.tolist()]
    else:
        modified_merged = pd.DataFrame()  # empty
        modified_keys = []

    return {
        'new_rows': new_rows,
        'deleted_rows': deleted_rows,
        'modified_merged': modified_merged,
        'modified_keys': modified_keys
    }

def make_diff_styler(merged_df: pd.DataFrame, pk_cols: List[str], view: str = None) -> pd.io.formats.style.Styler:
    """
    Given merged_df (output 'modified_merged' from compare_dataframes), produce a pandas Styler
    that highlights cells where old != new.

    view: None = full merged view, "old" = only old columns, "new" = only new columns
    """
    if merged_df is None or merged_df.empty:
        return pd.DataFrame().style

    # Build diff mask
    mask = pd.DataFrame(False, index=merged_df.index, columns=merged_df.columns)
    for base in [c[:-4] for c in merged_df.columns if c.endswith('_old')]:
        old_col = base + '_old'
        new_col = base + '_new'
        if old_col in merged_df.columns and new_col in merged_df.columns:
            neq = ~merged_df[old_col].combine(merged_df[new_col], lambda a, b: (pd.isna(a) and pd.isna(b)) or a == b)
            neq = neq.astype(bool)
            mask.loc[neq, old_col] = True
            mask.loc[neq, new_col] = True

    # Select columns for view
    if view == "old":
        df_view = merged_df[[c for c in merged_df.columns if c.endswith("_old") or c in pk_cols]].copy()
        mask = mask[df_view.columns]
    elif view == "new":
        df_view = merged_df[[c for c in merged_df.columns if c.endswith("_new") or c in pk_cols]].copy()
        mask = mask[df_view.columns]
    else:
        df_view = merged_df

    # Highlight
    def highlight_func(val, is_diff):
        return 'background-color: #ffef96' if is_diff else ''

    styler = df_view.style.apply(
        lambda row: [highlight_func(v, is_diff) for v, is_diff in zip(row, mask.loc[row.name])],
        axis=1
    )
    styler = styler.set_table_attributes('border="1" class="dataframe table table-striped"')
    return styler

def make_diff_html(merged_df: pd.DataFrame, pk_cols: List[str]) -> str:
    """
    Return an HTML table string where cells that changed (old vs new) are wrapped in
    <span style="background-color: #ffef96">value</span> for highlighting.

    This is handy for Streamlit where you can st.write(html, unsafe_allow_html=True).
    """
    if merged_df is None or merged_df.empty:
        return "<div>No modified rows</div>"

    df = merged_df.copy()

    # For each base col, create human-friendly presentation: Old vs New columns remain but values that differ are wrapped.
    base_cols = []
    for c in df.columns:
        if c.endswith('_old'):
            base_cols.append(c[:-4])

    # For safety, convert all to string for HTML rendering while preserving NaN as empty
    def to_display(val):
        if pd.isna(val):
            return ""
        return str(val)

    for base in base_cols:
        old_col = base + '_old'
        new_col = base + '_new'
        if old_col in df.columns and new_col in df.columns:
            # compute inequality mask
            neq = ~df[old_col].combine(df[new_col], lambda a, b: (pd.isna(a) and pd.isna(b)) or a == b).astype(bool)
            # wrap differing cells
            df[old_col] = [f'<span style="background-color:#ffd9a8">{to_display(v)}</span>' if neq_i else to_display(v) for v, neq_i in zip(df[old_col], neq)]
            df[new_col] = [f'<span style="background-color:#ffd9a8">{to_display(v)}</span>' if neq_i else to_display(v) for v, neq_i in zip(df[new_col], neq)]

    # Return HTML table
    return df.to_html(escape=False, index=False)
def split_old_new_tables(df: pd.DataFrame, pk_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split merged diff DataFrame into two DataFrames: old values and new values.
    Both keep PK columns for alignment.
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    old_df = pd.DataFrame()
    new_df = pd.DataFrame()

    # Keep PK columns first
    for pk in pk_cols:
        if pk in df.columns:
            old_df[pk] = df[pk]
            new_df[pk] = df[pk]

    # Fill old and new dataframes
    for col in df.columns:
        if col.endswith("_old"):
            base = col[:-4]
            old_df[base] = df[col]
        elif col.endswith("_new"):
            base = col[:-4]
            new_df[base] = df[col]

    return old_df, new_df

# Utility to allow CSV download creation for results
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.read()
