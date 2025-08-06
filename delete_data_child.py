"""Assumptions:
    - You pass a cursor from your DB connection.
    - You handle the commit and rollback logic outside this function if needed.
    - Your database supports foreign key introspection through standard information schema tables or error codes.

"""
from typing import List, Any, Union, Dict

def delete_record_safe(
    cursor,
    sql: str,
    params: List[Any] = None,
    output_format: str = 'text'
) -> Union[str, List[str], Dict[str, Any]]:
    """
    Attempts to delete a record from the database.
    If deletion fails due to foreign key constraints, returns a safe error message.

    Parameters:
    - cursor: Active DB-API 2.0 cursor (e.g., from psycopg2, sqlite3, etc.)
    - sql: Parameterized DELETE SQL statement (e.g., "DELETE FROM parent WHERE id = %s")
    - params: Parameters for the SQL query.
    - output_format: 'text' or 'list' - output format of error or success.

    Returns:
    - Success or error message, formatted according to output_format.
    """

    try:
        cursor.execute(sql, params or [])
        if cursor.rowcount == 0:
            result = "❌ No record found to delete."
        else:
            result = "✅ Record deleted successfully."

    except Exception as e:
        error_message = str(e).lower()
        if "foreign key" in error_message or "constraint" in error_message or "violates" in error_message:
            result = (
                "❌ Cannot delete record: Child records exist. "
                "Please delete dependent records first."
            )
        else:
            result = f"❌ Deletion failed: {str(e)}"

    if output_format == 'list':
        return [result]
    elif output_format == 'text':
        return result
    else:
        return {"status": "unknown_format", "message": result}
