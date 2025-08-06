# get_data.py
import pandas as pd
from typing import List, Union, Any, Dict
import psycopg2.extensions


def get_data(
    db_connection: psycopg2.extensions.connection,
    sql: str,
    parameters: List[Any] = None,
    output_type: str = 'dataframe'
) -> Union[pd.DataFrame, List[Dict]]:
    """
    Execute a parameterized SELECT query and return data in requested format.
    
    Args:
        db_connection: psycopg2 database connection object
        sql (str): SELECT SQL query with ? placeholders
        parameters (List[Any], optional): Parameter values for the placeholders
        output_type (str): 'dataframe' (default) or 'normal'
    
    Returns:
        Union[pd.DataFrame, List[Dict]]: Query result
    
    Raises:
        ValueError: For invalid SQL or parameter mismatch
        RuntimeError: For DB execution issues
    """

    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    if output_type.lower() not in ['dataframe', 'normal']:
        raise ValueError("output_type must be 'dataframe' or 'normal'")

    if parameters is None:
        parameters = []

    try:
        # Validate placeholder count
        placeholder_count = sql.count('?')
        if placeholder_count != len(parameters):
            raise ValueError(f"Parameter count mismatch: expected {placeholder_count}, got {len(parameters)}")

        # Convert ? to %s for psycopg2
        sql_converted = sql.replace('?', '%s')

        cursor = db_connection.cursor()
        cursor.execute(sql_converted, parameters)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        result = [dict(zip(columns, row)) for row in rows]

        if output_type.lower() == 'dataframe':
            return pd.DataFrame(result)
        else:
            return result

    except Exception as e:
        raise RuntimeError(f"Query execution error: {str(e)}")
