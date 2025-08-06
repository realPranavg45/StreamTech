import psycopg2

# PostgreSQL connection
db_connection = psycopg2.connect(
    dbname="testing",
    user="postgres",
    password="Pranav2004",
    host="localhost",
    port="5522"
)

def update_data(sql: str, parameters: list = None, output_type: str = "dict"):
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

    try:
        cursor = db_connection.cursor()
        cursor.execute(sql_clean.replace('?', '%s'), parameters)
        db_connection.commit()

        return _format_success(sql_clean, parameters, "UPDATE", output_type)

    except Exception as e:
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

# ✅ Run test
sql = "UPDATE test_users SET email = ? WHERE name = ?"
params = ["preddnaz@example.com", "John"]
result = update_data(sql, params, "list")  # <- OUTPUT FORMAT is passed here
print(result)
