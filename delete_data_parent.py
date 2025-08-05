import re
from typing import List, Any, Union

def safe_delete_record(cursor, delete_sql: str, params: List[Any], output_format: str = "text") -> Union[str, List[str]]:
    """
    Attempts to delete a record using the given DELETE query and parameters.
    If a foreign key constraint violation occurs, it identifies and deletes child records first.

    Assumptions:
    - You pass a cursor from your DB connection.
    - You handle the commit and rollback logic outside this function if needed.
    - Your database supports foreign key introspection through standard information schema tables or error codes.

    Args:
        cursor: An open database cursor object.
        delete_sql (str): The DELETE SQL query with placeholders.
        params (List[Any]): List of parameters for the DELETE query.
        output_format (str): "text" or "list" - the format of the returned message(s).

    Returns:
        Union[str, List[str]]: Outcome message(s).
    """
    messages = []

    def format_output(msg: str):
        print(msg)  # Debug output
        if output_format == "list":
            messages.append(msg)
        else:
            messages.append(msg + "\n")

    try:
        print("👉 Attempting to delete parent record")
        cursor.execute(delete_sql, params)
        format_output("✅ Record deleted successfully.")
        return messages if output_format == "list" else ''.join(messages)

    except Exception as e:
        error_msg = str(e)

        if "violates foreign key constraint" in error_msg or "foreign key constraint fails" in error_msg:
            cursor.connection.rollback()  # Reset the transaction before continuing
            format_output("⚠️ Foreign key constraint detected. Attempting to delete dependent child records...")

            fk_info = extract_fk_info(error_msg)

            if not fk_info:
                format_output("❌ Could not identify child records from the error message.")
                raise

            for child_table, foreign_key_column, parent_id in fk_info:
                child_delete_sql = f"DELETE FROM {child_table} WHERE {foreign_key_column} = %s"
                try:
                    print(f"👉 Deleting from {child_table} where {foreign_key_column} = {parent_id}")
                    cursor.execute(child_delete_sql, [parent_id])
                    format_output(f"🧹 Deleted child records from {child_table}.")
                except Exception as child_err:
                    format_output(f"❌ Failed to delete child records from {child_table}: {child_err}")
                    cursor.connection.rollback()
                    raise

            try:
                print("👉 Retrying parent delete after child cleanup")
                cursor.execute(delete_sql, params)
                format_output("✅ Parent record deleted successfully after cleaning child records.")
            except Exception as final_err:
                format_output(f"❌ Failed to delete parent after child cleanup: {final_err}")
                cursor.connection.rollback()
                raise

            return messages if output_format == "list" else ''.join(messages)

        else:
            format_output(f"❌ Error during deletion: {error_msg}")
            cursor.connection.rollback()
            raise


def extract_fk_info(error_msg: str) -> List[tuple]:
    """
    Parses a foreign key error message and attempts to extract child table and key info.
    NOTE: This is DB-specific and may need adjustment.

    Returns:
        A list of tuples: (child_table, foreign_key_column, parent_id)
    """
    fk_details = []

    # PostgreSQL-style FK error parsing
    match = re.search(r'Key \((.*?)\)=\((.*?)\) is still referenced from table "(.*?)"', error_msg)
    if match:
        fk_col = match.group(1)
        fk_val = match.group(2)
        child_table = match.group(3)
        fk_details.append((child_table, fk_col, fk_val))

    return fk_details
