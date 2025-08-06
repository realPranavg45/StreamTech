#import db_connection  # Make sure this is a connection instance, not a module

def update_data(sql: str, parameters: list = None, output_type: str = "dict"):
    """
    Executes only UPDATE SQL statements with ? placeholders and returns a structured response.

    Args:
        sql (str): SQL query string (e.g., "UPDATE table SET col = ? WHERE id = ?")
        parameters (list): List of values to substitute into query
        output_type (str): Output format - "dict" (default), "list", or "text"

    Returns:
        dict/list/str: Structured output depending on chosen format
    """

    if not sql or not sql.strip():
        return _format_error("SQL query is required", output_type)

    sql_clean = sql.strip()
    sql_lower = sql_clean.lower()

    if not sql_lower.startswith("update"):
        return _format_error("Only UPDATE operation is supported", output_type)

    if parameters is None:
        parameters = []

    placeholder_count = sql_clean.count('?')
    if len(parameters) != placeholder_count:
        return _format_error(
            f"Parameter count mismatch: expected {placeholder_count}, got {len(parameters)}",
            output_type
        )
#if connecting with databse then do uncomment this 
    #try:
        conn = db_connection  # ensure this is a valid DB connection instance
        cursor = conn.cursor()
        cursor.execute(sql_clean, parameters)
        conn.commit()

        return _format_success(sql_clean, parameters, "UPDATE", output_type)

    #except Exception as e:
        return _format_error(f"Processing error: {str(e)}", output_type)


def _format_error(msg, fmt):
    if fmt == "list":
        return ["error", msg]
    elif fmt == "text":
        return f"Error: {msg}"
    return {"status": "error", "message": msg}


def _format_success(sql, params, op, fmt):
    if fmt == "list":
        return ["success", sql, params, op]
    elif fmt == "text":
        return f"Status: success, Operation: {op}, SQL: {sql}, Parameters: {len(params)} values"
    return {
        "status": "success",
        "sql": sql,
        "parameters": params,
        "operation": op,
        "parameter_count": len(params)
    }
